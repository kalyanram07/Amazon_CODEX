"""
Stage 1: Detailed Validation Diagnostics & Error Analysis Engine.
Computes complete metric breakdown across countries, sources, and match cardinalities.
Outputs:
- Metric Breakdown Table (Overall, US, India, S2, S3, 0-match, 1-match, 2-5, 6+)
- Worst 1,000 validation error entities (experiments/worst_1000_validation_errors.csv)
"""

import os
import sys
import time
import json
import csv
import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
import pickle
from rapidfuzz import fuzz

sys.path.insert(0, os.path.abspath("."))

from src.preprocessing import preprocess_record
from src.blocking import CountrySoftIDFBlocker, benchmark_blocking_recall
from src.features import extract_pair_features, FEATURE_COLUMNS, EXPANDED_FEATURE_COLUMNS
from src.evaluate import compute_entity_f_beta


def run_stage1_diagnostics():
    print("=" * 80)
    print("           STAGE 1: COMPREHENSIVE VALIDATION DIAGNOSTIC AUDIT           ")
    print("=" * 80)
    t0 = time.time()

    dataset_dir = "dataset/train"
    s1_path = os.path.join(dataset_dir, "train_source1.tsv")
    s2_path = os.path.join(dataset_dir, "train_source2.tsv")
    s3_path = os.path.join(dataset_dir, "train_source3.tsv")
    gt_path = os.path.join(dataset_dir, "train_ground_truth.tsv")

    # 1. Load 14,999 Validation S1 records
    print("[1/5] Loading 14,999 Validation S1 records...", flush=True)
    val_s1_records: List[Dict[str, Any]] = []
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
                if 100000 <= count < 114999:
                    rec = preprocess_record(eid, name, addr, cntry)
                    val_s1_records.append(rec)
                    val_s1_by_country[cntry].append(rec)
                    val_s1_id_set.add(eid)
                if count >= 114999:
                    break

    print(f"Loaded {len(val_s1_records):,d} validation S1 records. Countries: { {c: len(v) for c, v in val_s1_by_country.items()} }", flush=True)

    # 2. Load Ground Truth
    print("[2/5] Loading Ground Truth...", flush=True)
    gt_map: Dict[str, Set[str]] = {}
    needed_targets: Set[str] = set()

    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            eid = parts[0]
            if eid in val_s1_id_set:
                matches = set(m.strip() for m in parts[1].split(",") if m.strip()) if len(parts) > 1 and parts[1].strip() else set()
                gt_map[eid] = matches
                needed_targets.update(matches)

    total_val_tp = sum(len(m) for m in gt_map.values())
    print(f"Validation set has {total_val_tp:,d} true target matches across {len(gt_map):,d} entities.", flush=True)

    # 3. Load Target Pool (all needed matches + background hard negatives)
    print("[3/5] Loading Target Pool for validation (needed matches + 200k background targets)...", flush=True)
    targets_by_country: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    targets_by_id: Dict[str, Dict[str, Any]] = {}
    country_bg_count = {c: 0 for c in val_s1_by_country.keys()}
    max_bg = 100000

    for target_path in [s2_path, s3_path]:
        with open(target_path, "r", encoding="utf-8") as tf:
            tf.readline()
            for line in tf:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    eid, name, addr, cntry = parts[0], parts[1], parts[2], parts[3].strip().upper()
                    if cntry in val_s1_by_country:
                        is_needed = eid in needed_targets
                        can_add_bg = country_bg_count[cntry] < max_bg
                        if is_needed or can_add_bg:
                            rec = preprocess_record(eid, name, addr, cntry)
                            targets_by_country[cntry].append(rec)
                            targets_by_id[eid] = rec
                            if not is_needed:
                                country_bg_count[cntry] += 1

    total_targets = sum(len(v) for v in targets_by_country.values())
    print(f"Loaded {total_targets:,d} target records across {list(targets_by_country.keys())}", flush=True)

    # 4. Load Best Model
    print("[4/5] Loading LightGBM Model artifact...", flush=True)
    model_path = "experiments/best_model/model/entity_resolution_model.pkl"
    if not os.path.exists(model_path):
        model_path = "models/entity_resolution_lgb.pkl"

    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    clf = artifact["model"]
    feat_cols = artifact.get("feature_columns", FEATURE_COLUMNS)
    tau_s = artifact.get("tau_singleton", 0.90)
    tau_m = artifact.get("tau_match", 0.80)
    print(f"Loaded model ({len(feat_cols)} features). Thresholds: tau_singleton={tau_s:.2f}, tau_match={tau_m:.2f}", flush=True)

    # 5. Fit Blocker & Run Inference
    print("[5/5] Running Blocker & Evaluating Detailed S1 Metrics...", flush=True)
    blockers: Dict[str, CountrySoftIDFBlocker] = {}
    for cntry, t_list in targets_by_country.items():
        blk = CountrySoftIDFBlocker(default_top_k=25, enable_char3=False, max_token_frequency=3000)
        blk.fit_records(t_list)
        blockers[cntry] = blk

    all_error_records = []
    entity_f05_list = []
    country_f05 = defaultdict(list)
    source_results = {"S2": {"tp": 0, "fp": 0, "fn": 0}, "S3": {"tp": 0, "fp": 0, "fn": 0}}
    cardinality_f05 = {"0_match": [], "1_match": [], "2_5_match": [], "6_plus_match": []}

    captured_tp_count = 0
    total_preds_count = 0
    total_tp_count = 0
    total_fp_count = 0
    singleton_correct = 0
    total_true_singletons = 0

    for s1 in val_s1_records:
        s1_id = s1["entity_id"]
        cntry = s1.get("country_norm", "").upper()
        true_matches = gt_map.get(s1_id, set())
        is_true_singleton = (len(true_matches) == 0)
        if is_true_singleton:
            total_true_singletons += 1

        blk = blockers.get(cntry)
        cands_with_scores = blk.block_entity_with_scores(s1, top_k=25) if blk else []
        cand_records = [c[0] for c in cands_with_scores]
        cand_ids = set(c["entity_id"] for c in cand_records)

        # Check blocker recall for this S1
        cand_tp = true_matches.intersection(cand_ids)
        captured_tp_count += len(cand_tp)
        all_true_in_candidates = (len(cand_tp) == len(true_matches)) if true_matches else True

        # Feature extraction & classification
        pred_matches = set()
        probs_with_cands = []

        if cand_records:
            feat_matrix = []
            for cand, blk_score, blk_rank in cands_with_scores:
                f_list = extract_pair_features(s1, cand)
                if len(feat_cols) > len(FEATURE_COLUMNS):
                    from src.features import extract_expanded_pair_features
                    f_list = extract_expanded_pair_features(s1, cand, blk_rank, blk_score)
                feat_matrix.append(f_list)

            X_mat = np.array(feat_matrix, dtype=np.float32)
            probs = clf.predict_proba(X_mat)[:, 1]

            for i, p in enumerate(probs):
                probs_with_cands.append((p, cand_records[i]))

            probs_with_cands.sort(key=lambda x: x[0], reverse=True)
            max_p = probs_with_cands[0][0]

            if max_p >= tau_s:
                for p, cand in probs_with_cands:
                    if p >= tau_m:
                        pred_matches.add(cand["entity_id"])

        # Metric evaluation for this S1
        tp = len(true_matches.intersection(pred_matches))
        fp = len(pred_matches - true_matches)
        fn = len(true_matches - pred_matches)

        total_tp_count += tp
        total_fp_count += fp
        total_preds_count += len(pred_matches)

        if is_true_singleton:
            if len(pred_matches) == 0:
                singleton_correct += 1
                f05 = 1.0
            else:
                f05 = 0.0
        else:
            if tp == 0:
                f05 = 0.0
            else:
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                beta_sq = 0.5 ** 2
                f05 = (1.0 + beta_sq) * (prec * rec) / ((beta_sq * prec) + rec) if (prec + rec) > 0 else 0.0

        entity_f05_list.append(f05)
        country_f05[cntry].append(f05)

        # Source breakdown
        for m_id in pred_matches:
            src = "S2" if m_id.startswith("S2") or "s2" in m_id.lower() else "S3"
            if m_id in true_matches:
                source_results[src]["tp"] += 1
            else:
                source_results[src]["fp"] += 1
        for m_id in (true_matches - pred_matches):
            src = "S2" if m_id.startswith("S2") or "s2" in m_id.lower() else "S3"
            source_results[src]["fn"] += 1

        # Match cardinality breakdown
        n_true = len(true_matches)
        if n_true == 0:
            cardinality_f05["0_match"].append(f05)
        elif n_true == 1:
            cardinality_f05["1_match"].append(f05)
        elif 2 <= n_true <= 5:
            cardinality_f05["2_5_match"].append(f05)
        else:
            cardinality_f05["6_plus_match"].append(f05)

        # Record error analysis if imperfect
        if f05 < 0.999:
            best_p = probs_with_cands[0][0] if probs_with_cands else 0.0
            second_p = probs_with_cands[1][0] if len(probs_with_cands) > 1 else 0.0
            best_cand = probs_with_cands[0][1] if probs_with_cands else None
            name_sim = fuzz.ratio(s1.get("name_std", ""), best_cand.get("name_std", "")) / 100.0 if best_cand else 0.0
            addr_sim = fuzz.ratio(s1.get("addr_std", ""), best_cand.get("addr_std", "")) / 100.0 if (best_cand and s1.get("addr_std")) else 0.0

            error_reason = "UNKNOWN"
            if is_true_singleton and len(pred_matches) > 0:
                error_reason = "FALSE_MERGE_ON_SINGLETON"
            elif not all_true_in_candidates:
                error_reason = "BLOCKER_MISSED_CANDIDATE"
            elif max_p < tau_s and not is_true_singleton:
                error_reason = "UNDERCONFIDENT_FALSE_SINGLETON"
            elif fp > 0 and tp > 0:
                error_reason = "OVER_PREDICTED_FALSE_POSITIVES"
            elif fn > 0:
                error_reason = "THRESHOLD_REJECTED_TRUE_MATCH"

            all_error_records.append({
                "s1_id": s1_id,
                "country": cntry,
                "true_matches": ",".join(sorted(list(true_matches))),
                "predicted_matches": ",".join(sorted(list(pred_matches))),
                "candidate_count": len(cand_records),
                "true_in_candidates": "YES" if all_true_in_candidates else "NO",
                "best_prob": round(float(best_p), 4),
                "second_best_prob": round(float(second_p), 4),
                "name_similarity": round(float(name_sim), 4),
                "address_similarity": round(float(addr_sim), 4),
                "f05_score": round(float(f05), 4),
                "error_reason": error_reason,
            })

    # Summary Calculations
    macro_f05 = float(np.mean(entity_f05_list))
    overall_prec = total_tp_count / (total_tp_count + total_fp_count) if (total_tp_count + total_fp_count) > 0 else 1.0
    overall_rec = total_tp_count / total_val_tp if total_val_tp > 0 else 1.0
    cand_recall = captured_tp_count / total_val_tp if total_val_tp > 0 else 1.0
    singleton_acc = singleton_correct / total_true_singletons if total_true_singletons > 0 else 1.0

    # Source F0.5
    s2_p = source_results["S2"]["tp"] / (source_results["S2"]["tp"] + source_results["S2"]["fp"]) if (source_results["S2"]["tp"] + source_results["S2"]["fp"]) > 0 else 1.0
    s2_r = source_results["S2"]["tp"] / (source_results["S2"]["tp"] + source_results["S2"]["fn"]) if (source_results["S2"]["tp"] + source_results["S2"]["fn"]) > 0 else 1.0
    s2_f05 = 1.25 * (s2_p * s2_r) / (0.25 * s2_p + s2_r) if (s2_p + s2_r) > 0 else 0.0

    s3_p = source_results["S3"]["tp"] / (source_results["S3"]["tp"] + source_results["S3"]["fp"]) if (source_results["S3"]["tp"] + source_results["S3"]["fp"]) > 0 else 1.0
    s3_r = source_results["S3"]["tp"] / (source_results["S3"]["tp"] + source_results["S3"]["fn"]) if (source_results["S3"]["tp"] + source_results["S3"]["fn"]) > 0 else 1.0
    s3_f05 = 1.25 * (s3_p * s3_r) / (0.25 * s3_p + s3_r) if (s3_p + s3_r) > 0 else 0.0

    # Print Formatted Results Table
    print("\n" + "=" * 80)
    print("                 STAGE 1: VALIDATION METRICS BREAKDOWN TABLE             ")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Score / Value':<20}")
    print("-" * 80)
    print(f"{'Overall Macro F0.5':<30} | {macro_f05:<20.6f}")
    print(f"{'Precision':<30} | {overall_prec:<20.4%}")
    print(f"{'Recall':<30} | {overall_rec:<20.4%}")
    print(f"{'Candidate Recall':<30} | {cand_recall:<20.4%}")
    print(f"{'Singleton Accuracy':<30} | {singleton_acc:<20.4%}")
    print(f"{'US Macro F0.5':<30} | {np.mean(country_f05['US']):<20.6f}")
    print(f"{'India Macro F0.5':<30} | {np.mean(country_f05['INDIA']):<20.6f}")
    print(f"{'Source 2 (S2) F0.5':<30} | {s2_f05:<20.6f}")
    print(f"{'Source 3 (S3) F0.5':<30} | {s3_f05:<20.6f}")
    print(f"{'0-Match (Singletons) F0.5':<30} | {np.mean(cardinality_f05['0_match']):<20.6f}")
    print(f"{'1-Match F0.5':<30} | {np.mean(cardinality_f05['1_match']):<20.6f}")
    print(f"{'2-5 Matches F0.5':<30} | {np.mean(cardinality_f05['2_5_match']):<20.6f}")
    print(f"{'6+ Matches F0.5':<30} | {np.mean(cardinality_f05['6_plus_match']):<20.6f}")
    print("=" * 80)

    # Save worst 1000 errors
    all_error_records.sort(key=lambda x: x["f05_score"])
    worst_1000 = all_error_records[:1000]
    out_dir = "experiments"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "worst_1000_validation_errors.csv")

    if worst_1000:
        keys = list(worst_1000[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(worst_1000)
        print(f"\n[SAVED] Worst 1,000 validation errors saved to: {out_csv}")

    # Root cause distribution
    reasons = defaultdict(int)
    for err in all_error_records:
        reasons[err["error_reason"]] += 1
    print("\nRoot Cause Breakdown Across All Imperfect Entities:")
    for r, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {r:<32}: {count:>5,d} entities ({count/len(all_error_records):>6.2%})")
    print(f"\nTotal Elapsed Time: {time.time() - t0:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    run_stage1_diagnostics()
