"""
Step 1: Exact Baseline Reproduction Script for Amazon ML Challenge 2026.
Reproduces the verified baseline on the 15,000 S1 holdout split and saves full metrics to
experiments/soft_idf_v1/baseline_metrics.json.
"""

import os
import sys
import time
import json
import pickle
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from src.preprocessing import preprocess_record
from src.blocking import CountryBlocker, benchmark_blocking_recall
from src.features import extract_pair_features, FEATURE_COLUMNS
from src.evaluate import evaluate, print_evaluation_report
import lightgbm as lgb


def run_baseline_reproduction(
    dataset_dir: str = "dataset",
    output_dir: str = "experiments/soft_idf_v1",
    max_s1_train: int = 100000,
    val_size: int = 15000,
    random_state: int = 42,
) -> Dict[str, Any]:
    print("=" * 65)
    print("      STEP 1: REPRODUCING VERIFIED BASELINE METRICS      ")
    print("=" * 65)
    t_start = time.time()

    train_dir = os.path.join(dataset_dir, "train")
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    # 1. Load S1
    print(f"Loading top {max_s1_train + val_size:,d} Source 1 records...", flush=True)
    s1_by_country: Dict[str, List[Dict[str, Any]]] = {}
    s1_id_set: Set[str] = set()
    count_s1 = 0

    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                eid, name, addr, cntry = parts[0], parts[1], parts[2], parts[3].strip().upper()
                rec = preprocess_record(eid, name, addr, cntry)
                s1_by_country.setdefault(cntry, []).append(rec)
                s1_id_set.add(eid)
                count_s1 += 1
                if count_s1 >= (max_s1_train + val_size):
                    break

    # 2. Load GT
    gt_map: Dict[str, Set[str]] = {}
    needed_targets: Set[str] = set()
    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            eid = parts[0]
            if eid in s1_id_set:
                matches = set(m.strip() for m in parts[1].split(",") if m.strip()) if len(parts) > 1 and parts[1].strip() else set()
                gt_map[eid] = matches
                needed_targets.update(matches)

    # 3. Load Targets
    background_targets_per_country = 300000
    targets_by_country: Dict[str, List[Dict[str, Any]]] = {}
    country_bg_count = {c: 0 for c in s1_by_country.keys()}

    for target_path in [s2_path, s3_path]:
        with open(target_path, "r", encoding="utf-8") as tf:
            tf.readline()
            for line in tf:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    eid, name, addr, cntry = parts[0], parts[1], parts[2], parts[3].strip().upper()
                    if cntry in s1_by_country:
                        is_needed = eid in needed_targets
                        can_add_bg = country_bg_count[cntry] < background_targets_per_country
                        if is_needed or can_add_bg:
                            rec = preprocess_record(eid, name, addr, cntry)
                            targets_by_country.setdefault(cntry, []).append(rec)
                            if not is_needed:
                                country_bg_count[cntry] += 1

    # 4. Train / Val Split (Grouped by S1 entities with random_state=42)
    all_train_s1 = []
    all_val_s1 = []
    np.random.seed(random_state)

    for country, s1_list in s1_by_country.items():
        n_val = int(len(s1_list) * (val_size / (max_s1_train + val_size)))
        indices = np.random.permutation(len(s1_list))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        all_val_s1.extend([s1_list[i] for i in val_idx])
        all_train_s1.extend([s1_list[i] for i in train_idx])

    val_s1_subset = all_val_s1[:val_size]
    print(f"Split: {len(all_train_s1):,d} Train S1 | {len(val_s1_subset):,d} Validation S1", flush=True)

    # 5. Fit baseline blockers
    val_blockers = {}
    X_train_list = []
    y_train_list = []

    for country in s1_by_country.keys():
        country_targets = targets_by_country.get(country, [])
        blocker = CountryBlocker(max_token_frequency=2500, default_top_k=25).fit_records(country_targets)
        val_blockers[country] = blocker
        targets_by_id = {t["entity_id"]: t for t in country_targets}
        country_train_s1 = [s for s in all_train_s1 if s["country_norm"] == country.lower()]

        # Build train pairs
        X_rows = []
        y_rows = []
        pos_count = 0
        neg_count = 0
        for s1 in country_train_s1:
            s1_id = s1["entity_id"]
            true_m = gt_map.get(s1_id, set())
            for m_id in true_m:
                if m_id in targets_by_id and pos_count < 100000:
                    X_rows.append(extract_pair_features(s1, targets_by_id[m_id]))
                    y_rows.append(1)
                    pos_count += 1
            for cand in blocker.block_entity(s1, top_k=20):
                c_id = cand["entity_id"]
                if c_id not in true_m and neg_count < 100000:
                    X_rows.append(extract_pair_features(s1, cand))
                    y_rows.append(0)
                    neg_count += 1
            if pos_count >= 100000 and neg_count >= 100000:
                break
        if X_rows:
            X_train_list.append(np.array(X_rows, dtype=np.float32))
            y_train_list.append(np.array(y_rows, dtype=np.int32))

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    print(f"Training dataset: {len(X_train):,d} pairs ({sum(y_train==1):,d} pos, {sum(y_train==0):,d} neg)", flush=True)

    # 6. Fit LightGBM Model
    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.08,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # 7. Evaluate on Validation S1 entities
    val_gt = {s1["entity_id"]: gt_map.get(s1["entity_id"], set()) for s1 in val_s1_subset}
    cand_map: Dict[str, Set[str]] = {}
    preds_map: Dict[str, Set[str]] = {}
    cands_per_s1_list = []

    tau_singleton = 0.80
    tau_match = 0.70

    for s1 in val_s1_subset:
        s1_id = s1["entity_id"]
        cntry = s1["country_norm"].upper()
        blocker = val_blockers.get(cntry)
        if not blocker:
            cand_map[s1_id] = set()
            preds_map[s1_id] = set()
            cands_per_s1_list.append(0)
            continue

        cands = blocker.block_entity(s1, top_k=25)
        cand_ids = [c["entity_id"] for c in cands]
        cand_map[s1_id] = set(cand_ids)
        cands_per_s1_list.append(len(cand_ids))

        if not cands:
            preds_map[s1_id] = set()
            continue

        feat_mat = np.array([extract_pair_features(s1, c) for c in cands], dtype=np.float32)
        probs = clf.predict_proba(feat_mat)[:, 1]

        if np.max(probs) < tau_singleton:
            preds_map[s1_id] = set()
        else:
            preds_map[s1_id] = {cand_ids[i] for i, p in enumerate(probs) if p >= tau_match}

    # Evaluate Report
    report = evaluate(val_gt, preds_map, candidate_pairs=cand_map)
    print_evaluation_report(report)

    # Compute candidate recall
    total_val_tp = sum(len(m) for m in val_gt.values())
    captured_val_tp = sum(len(val_gt[s1_id].intersection(cand_map[s1_id])) for s1_id in val_gt)
    cand_recall = captured_val_tp / total_val_tp if total_val_tp > 0 else 1.0

    # False positives and False negatives
    total_fp = sum(len(preds_map[s1_id] - val_gt[s1_id]) for s1_id in val_gt)
    total_fn = sum(len(val_gt[s1_id] - preds_map[s1_id]) for s1_id in val_gt)
    total_tp_matched = sum(len(val_gt[s1_id].intersection(preds_map[s1_id])) for s1_id in val_gt)

    runtime_s = round(time.time() - t_start, 2)

    baseline_metrics = {
        "experiment_name": "Baseline Reproduction",
        "macro_f0.5": report["macro_f0.5"],
        "precision": report["macro_precision"],
        "recall": report["macro_recall"],
        "singleton_accuracy": report["singleton_accuracy"],
        "candidate_recall": round(cand_recall, 6),
        "total_validation_s1": len(val_s1_subset),
        "total_true_pairs": total_val_tp,
        "captured_candidate_pairs": captured_val_tp,
        "true_positives": total_tp_matched,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "singletons_total": report["singleton_count"],
        "singletons_correct": report["correct_singletons"],
        "singletons_contaminated": report["contaminated_singletons"],
        "avg_candidates_per_s1": round(float(np.mean(cands_per_s1_list)), 2),
        "median_candidates_per_s1": float(np.median(cands_per_s1_list)),
        "max_candidates_per_s1": int(np.max(cands_per_s1_list)),
        "tau_singleton": tau_singleton,
        "tau_match": tau_match,
        "runtime_seconds": runtime_s,
    }

    out_json = os.path.join(output_dir, "baseline_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(baseline_metrics, f, indent=2)
    print(f"\n[SAVED] Baseline metrics saved to: {out_json}")

    return baseline_metrics


if __name__ == "__main__":
    run_baseline_reproduction()
