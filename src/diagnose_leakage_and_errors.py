"""
Stage 1: High-Speed Country-Partitioned Diagnostic Audit Script.
Evaluates the baseline model against the FULL unconstrained target dataset country-by-country.
Computes:
- Full target pool candidate recall
- Country-level F0.5 (US vs India)
- Source-level F0.5 (S2 vs S3)
- Match-cardinality F0.5 (0-match singletons, 1-match, 2-5 matches, 6+ matches)
- Generates worst_1000_validation_errors.csv with exact root-cause categorization
"""

import os
import sys
import time
import json
import csv
from collections import defaultdict
from array import array
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
import pickle

sys.path.insert(0, os.path.abspath("."))

from src.preprocessing import preprocess_record, clean_text_fast
from src.features import extract_pair_features, FEATURE_COLUMNS
from src.evaluate import compute_entity_f_beta


def run_country_diagnostic(
    country: str,
    s1_records: List[Dict[str, Any]],
    s2_path: str,
    s3_path: str,
    gt_map: Dict[str, Set[str]],
    clf: Any,
    tau_s: float,
    tau_m: float,
):
    print(f"\n--- Diagnostic for Country: {country} ({len(s1_records):,d} S1 entities) ---", flush=True)
    t0 = time.time()

    # Index targets for this country only
    target_records = []
    name_idx = defaultdict(lambda: array("i"))
    prefix_idx = defaultdict(lambda: array("i"))
    digit_idx = defaultdict(lambda: array("i"))
    addr_token_idx = defaultdict(lambda: array("i"))

    idx = 0
    for target_path in [s2_path, s3_path]:
        with open(target_path, "r", encoding="utf-8") as tf:
            tf.readline()
            for line in tf:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    eid, name, addr, cntry = parts[0], parts[1], parts[2], parts[3].strip().upper()
                    if cntry == country:
                        rec = preprocess_record(eid, name, addr, country)
                        target_records.append(rec)

                        for tok in rec.get("name_tokens", []):
                            if len(tok) >= 2:
                                name_idx[tok].append(idx)
                        pref = rec.get("name_prefix", "")
                        if pref:
                            prefix_idx[pref].append(idx)
                        for dig in rec.get("addr_digits", []):
                            digit_idx[dig].append(idx)
                        for tok in rec.get("addr_tokens", []):
                            if len(tok) >= 4:
                                addr_token_idx[tok].append(idx)
                        idx += 1

    # Prune high frequency keys
    for idx_dict in [name_idx, prefix_idx, digit_idx, addr_token_idx]:
        keys_to_del = [k for k, v in idx_dict.items() if len(v) > 3000]
        for k in keys_to_del:
            del idx_dict[k]

    print(f"[{country}] Loaded and indexed {len(target_records):,d} targets in {time.time() - t0:.2f}s", flush=True)

    # Evaluate S1 entities
    results = []
    error_records = []
    captured_tp = 0
    total_val_tp = 0
    source_metrics = {"S2": {"tp": 0, "fp": 0, "fn": 0}, "S3": {"tp": 0, "fp": 0, "fn": 0}}
    cardinality_metrics = defaultdict(list)

    for s1 in s1_records:
        s1_id = s1["entity_id"]
        true_m = gt_map.get(s1_id, set())
        total_val_tp += len(true_m)

        scores = defaultdict(float)
        for tok in s1.get("name_tokens", []):
            if tok in name_idx:
                for tid in name_idx[tok]:
                    scores[tid] += 3.0
        pref = s1.get("name_prefix", "")
        if pref and pref in prefix_idx:
            for tid in prefix_idx[pref]:
                scores[tid] += 4.0
        for dig in s1.get("addr_digits", []):
            if dig in digit_idx:
                for tid in digit_idx[dig]:
                    scores[tid] += 2.0
        for tok in s1.get("addr_tokens", []):
            if tok in addr_token_idx:
                for tid in addr_token_idx[tok]:
                    scores[tid] += 1.0

        if len(scores) > 25:
            top_tids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:25]
        else:
            top_tids = list(scores.keys())

        cands = [target_records[t] for t in top_tids]
        cand_ids = [c["entity_id"] for c in cands]
        cand_set = set(cand_ids)

        captured_tp += len(true_m.intersection(cand_set))

        if not cands:
            pred_m = set()
            probs = np.array([])
        else:
            feat_mat = np.array([extract_pair_features(s1, c) for c in cands], dtype=np.float32)
            probs = clf.predict_proba(feat_mat)[:, 1]
            if np.max(probs) < tau_s:
                pred_m = set()
            else:
                pred_m = {cand_ids[i] for i, p in enumerate(probs) if p >= tau_m}

        p, r, f05 = compute_entity_f_beta(true_m, pred_m, beta=0.5)
        results.append({"s1_id": s1_id, "f05": f05, "precision": p, "recall": r})

        # Source breakdown
        for mid in true_m:
            src = "S2" if mid.startswith("S2") else "S3"
            if mid in pred_m:
                source_metrics[src]["tp"] += 1
            else:
                source_metrics[src]["fn"] += 1
        for pid in pred_m:
            if pid not in true_m:
                src = "S2" if pid.startswith("S2") else "S3"
                source_metrics[src]["fp"] += 1

        # Cardinality breakdown
        num_true = len(true_m)
        if num_true == 0:
            card_key = "0_matches (Singletons)"
        elif num_true == 1:
            card_key = "1_match"
        elif 2 <= num_true <= 5:
            card_key = "2-5_matches"
        else:
            card_key = "6+_matches"
        cardinality_metrics[card_key].append(f05)

        if f05 < 0.90:
            best_prob = float(np.max(probs)) if len(probs) > 0 else 0.0
            second_prob = float(sorted(probs, reverse=True)[1]) if len(probs) > 1 else 0.0
            error_records.append({
                "s1_id": s1_id,
                "s1_name": s1["name_raw"],
                "s1_address": s1["addr_raw"],
                "country": country,
                "true_matches": ",".join(sorted(true_m)),
                "pred_matches": ",".join(sorted(pred_m)),
                "f05_score": round(f05, 4),
                "cand_count": len(cands),
                "true_match_in_candidates": "YES" if bool(true_m.intersection(cand_set)) else "NO",
                "best_predicted_prob": round(best_prob, 4),
                "second_best_prob": round(second_prob, 4),
                "error_reason": "BLOCKER_MISS" if not true_m.intersection(cand_set) and len(true_m) > 0 else (
                    "SINGLETON_FALSE_MERGE" if len(true_m) == 0 and len(pred_m) > 0 else "MODEL_THRESHOLD_OR_RANK_ERROR"
                )
            })

    cand_recall = captured_tp / total_val_tp if total_val_tp > 0 else 1.0
    country_res = {
        "country": country,
        "macro_f05": float(np.mean([r["f05"] for r in results])),
        "precision": float(np.mean([r["precision"] for r in results])),
        "recall": float(np.mean([r["recall"] for r in results])),
        "candidate_recall": float(cand_recall),
        "source_metrics": source_metrics,
        "cardinality_metrics": {k: float(np.mean(v)) for k, v in cardinality_metrics.items()},
        "error_records": error_records,
    }

    print(f"[{country}] Macro F0.5 = {country_res['macro_f05']:.4f} | Prec = {country_res['precision']:.4f} | Rec = {country_res['recall']:.4f} | Blocker Recall = {cand_recall:.2%}")
    return country_res


def run_full_diagnosis():
    print("=" * 75)
    print("      STAGE 1: COUNTRY-STREAMING FULL-SCALE DIAGNOSIS      ")
    print("=" * 75)

    dataset_dir = "dataset/train"
    s1_path = os.path.join(dataset_dir, "train_source1.tsv")
    s2_path = os.path.join(dataset_dir, "train_source2.tsv")
    s3_path = os.path.join(dataset_dir, "train_source3.tsv")
    gt_path = os.path.join(dataset_dir, "train_ground_truth.tsv")

    # Load 5,000 S1 validation records grouped by country
    val_s1_by_country: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    val_s1_id_set: Set[str] = set()

    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
        count = 0
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                eid, name, addr, cntry = parts[0], parts[1], parts[2], parts[3].strip().upper()
                count += 1
                if 100000 <= count < 105000:
                    rec = preprocess_record(eid, name, addr, cntry)
                    val_s1_by_country[cntry].append(rec)
                    val_s1_id_set.add(eid)
                if count >= 105000:
                    break

    # Load GT
    gt_map: Dict[str, Set[str]] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            eid = parts[0]
            if eid in val_s1_id_set:
                matches = set(m.strip() for m in parts[1].split(",") if m.strip()) if len(parts) > 1 and parts[1].strip() else set()
                gt_map[eid] = matches

    # Load Model
    model_path = "experiments/best_model/model/entity_resolution_model.pkl"
    if not os.path.exists(model_path):
        model_path = "models/entity_resolution_lgb.pkl"
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    clf = artifact["model"]
    tau_s = artifact.get("tau_singleton", 0.90)
    tau_m = artifact.get("tau_match", 0.80)

    country_reports = []
    all_errors = []

    for country in ["US", "INDIA"]:
        if country in val_s1_by_country:
            rep = run_country_diagnostic(
                country=country,
                s1_records=val_s1_by_country[country],
                s2_path=s2_path,
                s3_path=s3_path,
                gt_map=gt_map,
                clf=clf,
                tau_s=tau_s,
                tau_m=tau_m,
            )
            country_reports.append(rep)
            all_errors.extend(rep["error_records"])

    # Overall Summary
    overall_f05 = np.mean([r["macro_f05"] for r in country_reports])
    overall_prec = np.mean([r["precision"] for r in country_reports])
    overall_rec = np.mean([r["recall"] for r in country_reports])
    overall_cand_rec = np.mean([r["candidate_recall"] for r in country_reports])

    print("\n" + "=" * 75)
    print("                    OVERALL DIAGNOSTIC SUMMARY")
    print("=" * 75)
    print(f"Overall Full-Corpus Macro F0.5 : {overall_f05:.6f}")
    print(f"Overall Full-Corpus Precision  : {overall_prec:.6f}")
    print(f"Overall Full-Corpus Recall     : {overall_rec:.6f}")
    print(f"Overall Candidate Recall       : {overall_cand_rec:.2%}")
    print("-" * 75)

    # Save worst 1000
    all_errors.sort(key=lambda x: x["f05_score"])
    worst_1000 = all_errors[:1000]
    out_csv = "experiments/worst_1000_validation_errors.csv"
    if worst_1000:
        keys = list(worst_1000[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(worst_1000)
        print(f"[SAVED] Worst 1,000 validation errors saved to: {out_csv}")

    # Breakdown of error causes
    reasons = defaultdict(int)
    for err in all_errors:
        reasons[err["error_reason"]] += 1
    print("\nRoot Cause Error Breakdown:")
    for r, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {r:30s}: {count:,d} entities ({count/len(all_errors):.2%})")
    print("=" * 75)


if __name__ == "__main__":
    run_full_diagnosis()
