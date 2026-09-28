"""
Stage 2-6 Controlled Experiment:
- MultiChannelUnionBlocker (7-Channel Union Retrieval)
- 41 Expanded Pairwise Features (Jaro-Winkler, Char 3/4 cosine, Containment Ratios, Quadrants)
- Multi-Type Hard-Negative Mining (Same-Name/Diff-Addr, Same-Addr/Diff-Name, Blocker Top-K)
- S1 Margin & Dual Threshold Tuning on 14,999 Validation Split
"""

import os
import sys
import time
import json
import csv
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
from rapidfuzz import fuzz

sys.path.insert(0, os.path.abspath("."))

from src.preprocessing import preprocess_record
from src.blocking import MultiChannelUnionBlocker, benchmark_blocking_recall
from src.features import extract_expanded_pair_features, EXPANDED_FEATURE_COLUMNS
from src.evaluate import evaluate, print_evaluation_report


def run_training_experiment():
    print("=" * 80)
    print("      EXPERIMENT: 7-CHANNEL UNION BLOCKER + 41 ADVANCED FEATURES + HARD NEGATIVES      ")
    print("=" * 80)
    t_start = time.time()

    dataset_dir = "dataset/train"
    s1_path = os.path.join(dataset_dir, "train_source1.tsv")
    s2_path = os.path.join(dataset_dir, "train_source2.tsv")
    s3_path = os.path.join(dataset_dir, "train_source3.tsv")
    gt_path = os.path.join(dataset_dir, "train_ground_truth.tsv")

    # 1. Load S1 Records (100k Train + 14,999 Validation)
    print("[1/6] Loading Source 1 records...", flush=True)
    train_s1_records: List[Dict[str, Any]] = []
    val_s1_records: List[Dict[str, Any]] = []
    val_s1_by_country: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    all_s1_ids: Set[str] = set()

    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
        count = 0
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                eid, name, addr, cntry = parts[0], parts[1], parts[2], parts[3].strip().upper()
                count += 1
                rec = preprocess_record(eid, name, addr, cntry)
                all_s1_ids.add(eid)
                if count <= 100000:
                    train_s1_records.append(rec)
                elif count <= 114999:
                    val_s1_records.append(rec)
                    val_s1_by_country[cntry].append(rec)
                if count >= 114999:
                    break

    print(f"Loaded {len(train_s1_records):,d} Train S1s and {len(val_s1_records):,d} Validation S1s.", flush=True)

    # 2. Load Ground Truth
    print("[2/6] Loading Ground Truth...", flush=True)
    gt_map: Dict[str, Set[str]] = {}
    needed_targets: Set[str] = set()

    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            eid = parts[0]
            if eid in all_s1_ids:
                matches = set(m.strip() for m in parts[1].split(",") if m.strip()) if len(parts) > 1 and parts[1].strip() else set()
                gt_map[eid] = matches
                needed_targets.update(matches)

    # 3. Load Targets & Build Background Pool
    print("[3/6] Loading Targets (Needed matches + 250k Background Records)...", flush=True)
    targets_by_country: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    targets_by_id: Dict[str, Dict[str, Any]] = {}
    country_bg_count = {"US": 0, "INDIA": 0}

    for target_path in [s2_path, s3_path]:
        with open(target_path, "r", encoding="utf-8") as tf:
            tf.readline()
            for line in tf:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    eid, name, addr, cntry = parts[0], parts[1], parts[2], parts[3].strip().upper()
                    if cntry in country_bg_count:
                        is_needed = eid in needed_targets
                        if is_needed or country_bg_count[cntry] < 125000:
                            rec = preprocess_record(eid, name, addr, cntry)
                            targets_by_country[cntry].append(rec)
                            targets_by_id[eid] = rec
                            if not is_needed:
                                country_bg_count[cntry] += 1

    print(f"Loaded {len(targets_by_id):,d} targets across {list(targets_by_country.keys())}", flush=True)

    # 4. Fit MultiChannelUnionBlocker on Targets
    print("[4/6] Fitting MultiChannelUnionBlocker on target pool...", flush=True)
    blockers: Dict[str, MultiChannelUnionBlocker] = {}
    for cntry, t_list in targets_by_country.items():
        blk = MultiChannelUnionBlocker(default_top_k=35, max_token_frequency=4000, enable_char_ngrams=True)
        blk.fit_records(t_list)
        blockers[cntry] = blk

    # 5. Mine Training Pairs with Multi-Type Hard Negatives
    print("[5/6] Mining Positive & Multi-Type Hard-Negative pairs from blocker candidates...", flush=True)
    X_rows = []
    y_list = []
    pos_count = 0
    neg_count = 0
    max_pairs = 250000

    t_mine = time.time()
    for s1 in train_s1_records:
        s1_id = s1["entity_id"]
        cntry = s1.get("country_norm", "").upper()
        true_matches = gt_map.get(s1_id, set())

        blk = blockers.get(cntry)
        if not blk:
            continue

        cands_with_scores = blk.block_entity_with_scores(s1, top_k=25)
        for cand, score, rank in cands_with_scores:
            c_id = cand["entity_id"]
            is_pos = (c_id in true_matches)

            if is_pos and pos_count < max_pairs:
                feats = extract_expanded_pair_features(s1, cand, blocker_rank=rank, blocker_score=score)
                X_rows.append(feats)
                y_list.append(1)
                pos_count += 1
            elif not is_pos and neg_count < max_pairs:
                feats = extract_expanded_pair_features(s1, cand, blocker_rank=rank, blocker_score=score)
                X_rows.append(feats)
                y_list.append(0)
                neg_count += 1

        if pos_count >= max_pairs and neg_count >= max_pairs:
            break

    print(f"Constructed {len(X_rows):,d} training pairs in {time.time() - t_mine:.2f}s | Pos: {pos_count:,d} | Neg: {neg_count:,d}", flush=True)
    X_train = np.array(X_rows, dtype=np.float32)
    y_train = np.array(y_list, dtype=np.int32)

    # 6. Train LightGBM Classifier
    print("\n[6/6] Training LightGBM Classifier (41 features)...", flush=True)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "learning_rate": 0.04,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 25,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_train, y_train)

    # 7. Evaluate & Tune Thresholds on 14,999 Validation Entities
    print("\n--- Tuning Dual Thresholds on 14,999 Holdout Validation Entities ---", flush=True)
    val_gt = {s1["entity_id"]: gt_map.get(s1["entity_id"], set()) for s1 in val_s1_records}
    s1_val_eval_data = []

    t_val = time.time()
    for s1 in val_s1_records:
        s1_id = s1["entity_id"]
        cntry = s1.get("country_norm", "").upper()
        blk = blockers.get(cntry)
        if not blk:
            s1_val_eval_data.append((s1_id, [], []))
            continue

        cands_with_scores = blk.block_entity_with_scores(s1, top_k=35)
        if not cands_with_scores:
            s1_val_eval_data.append((s1_id, [], []))
            continue

        cand_ids = [c[0]["entity_id"] for c in cands_with_scores]
        feat_mat = np.array([
            extract_expanded_pair_features(s1, c[0], blocker_rank=c[2], blocker_score=c[1])
            for c in cands_with_scores
        ], dtype=np.float32)

        probs = clf.predict_proba(feat_mat)[:, 1]
        s1_val_eval_data.append((s1_id, cand_ids, probs))

    print(f"Validation inference completed in {time.time() - t_val:.2f}s", flush=True)

    # Grid search thresholds for Macro F0.5
    best_f05 = -1.0
    best_tau_s = 0.80
    best_tau_m = 0.70

    for tau_s in [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92]:
        for tau_m in [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
            preds = {}
            for s1_id, c_ids, probs in s1_val_eval_data:
                if not c_ids or len(probs) == 0:
                    preds[s1_id] = set()
                    continue
                max_p = np.max(probs)
                if max_p < tau_s:
                    preds[s1_id] = set()
                else:
                    matched = {c_ids[i] for i, p in enumerate(probs) if p >= tau_m}
                    preds[s1_id] = matched

            report = evaluate(val_gt, preds)
            f05 = report["macro_f0.5"]
            if f05 > best_f05:
                best_f05 = f05
                best_tau_s = tau_s
                best_tau_m = tau_m

    print("\n" + "=" * 80)
    print("                     EXPERIMENT VALIDATION RESULTS                     ")
    print("=" * 80)
    print(f"Optimal tau_singleton : {best_tau_s:.2f}")
    print(f"Optimal tau_match     : {best_tau_m:.2f}")
    print(f"Best Macro F0.5       : {best_f05:.6f}")

    # Generate full report with optimal thresholds
    best_preds = {}
    best_cands = {}
    for s1_id, c_ids, probs in s1_val_eval_data:
        best_cands[s1_id] = set(c_ids)
        if not c_ids or len(probs) == 0 or np.max(probs) < best_tau_s:
            best_preds[s1_id] = set()
        else:
            best_preds[s1_id] = {c_ids[i] for i, p in enumerate(probs) if p >= best_tau_m}

    final_report = evaluate(val_gt, best_preds, candidate_pairs=best_cands)
    print_evaluation_report(final_report)

    # Save experiment model & metrics
    exp_dir = "experiments/exp_union_expanded"
    os.makedirs(exp_dir, exist_ok=True)
    model_save_path = os.path.join(exp_dir, "entity_resolution_model.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump({
            "model": clf,
            "feature_columns": EXPANDED_FEATURE_COLUMNS,
            "tau_singleton": best_tau_s,
            "tau_match": best_tau_m,
            "macro_f05": best_f05,
            "top_k": 35,
        }, f)

    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "macro_f05": final_report["macro_f0.5"],
            "macro_precision": final_report["macro_precision"],
            "macro_recall": final_report["macro_recall"],
            "candidate_recall": final_report.get("candidate_recall", 0.0),
            "singleton_accuracy": final_report.get("singleton_accuracy", 0.0),
            "tau_singleton": best_tau_s,
            "tau_match": best_tau_m,
            "features_count": len(EXPANDED_FEATURE_COLUMNS),
        }, f, indent=2)

    print(f"[SAVED] Experiment model and metrics saved to: {exp_dir}")
    print(f"Total Experiment Runtime: {time.time() - t_start:.2f}s")
    print("=" * 80)

    print(f"[SAVED] Experiment model and metrics saved to: {exp_dir}")
    print(f"Total Experiment Runtime: {time.time() - t_start:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    run_training_experiment()
