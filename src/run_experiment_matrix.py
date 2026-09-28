"""
Master Experiment Matrix Runner for Amazon ML Challenge 2026:
Runs Experiments A through G sequentially on the EXACT same 14,999 holdout S1 entities,
records all metrics into experiments/results.csv, and finds the highest Macro F0.5 configuration.
"""

import os
import sys
import time
import json
import csv
import pickle
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.abspath("."))

from src.preprocessing import preprocess_record
from src.blocking import CountryBlocker, CountrySoftIDFBlocker
from src.features import (
    extract_pair_features,
    extract_expanded_pair_features,
    FEATURE_COLUMNS,
    EXPANDED_FEATURE_COLUMNS,
)
from src.evaluate import evaluate, print_evaluation_report


def load_data_and_split(
    train_dir: str = "dataset/train",
    max_s1_train: int = 100000,
    val_size: int = 15000,
    background_targets_per_country: int = 300000,
    random_state: int = 42,
):
    """Loads records and prepares consistent S1-grouped Train/Val split."""
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

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
    print(f"Loaded and partitioned: {len(all_train_s1):,d} Train S1 | {len(val_s1_subset):,d} Validation S1", flush=True)

    return s1_by_country, targets_by_country, gt_map, all_train_s1, val_s1_subset


def execute_experiment(
    exp_id: str,
    exp_name: str,
    blocker_type: str,  # 'baseline', 'soft_idf', 'soft_idf_char3'
    top_k: int,
    use_expanded_features: bool,
    mine_hard_negatives: bool,
    tau_singleton: float,
    tau_match: float,
    s1_by_country: Dict[str, List[Dict[str, Any]]],
    targets_by_country: Dict[str, List[Dict[str, Any]]],
    gt_map: Dict[str, Set[str]],
    all_train_s1: List[Dict[str, Any]],
    val_s1_subset: List[Dict[str, Any]],
    random_state: int = 42,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"  RUNNING {exp_id}: {exp_name} (Top-K={top_k}, ExpandedFeats={use_expanded_features}, HardNegs={mine_hard_negatives})")
    print("=" * 70, flush=True)
    t_start = time.time()

    # 1. Fit Blockers
    blockers = {}
    for country, country_targets in targets_by_country.items():
        if blocker_type == "baseline":
            b = CountryBlocker(max_token_frequency=2500, default_top_k=top_k).fit_records(country_targets)
        elif blocker_type == "soft_idf":
            b = CountrySoftIDFBlocker(default_top_k=top_k, enable_char3=False).fit_records(country_targets)
        elif blocker_type == "soft_idf_char3":
            b = CountrySoftIDFBlocker(default_top_k=top_k, enable_char3=True).fit_records(country_targets)
        else:
            raise ValueError(f"Unknown blocker type: {blocker_type}")
        blockers[country] = b

    # 2. Build Training Pairs
    X_train_list = []
    y_train_list = []

    for country, country_targets in targets_by_country.items():
        blocker = blockers[country]
        targets_by_id = {t["entity_id"]: t for t in country_targets}
        country_train_s1 = [s for s in all_train_s1 if s["country_norm"] == country.lower()]
        name_idf = getattr(blocker, "name_idf", None)

        X_rows = []
        y_rows = []
        pos_count = 0
        neg_count = 0

        for s1 in country_train_s1:
            s1_id = s1["entity_id"]
            true_m = gt_map.get(s1_id, set())

            # Positives
            for m_id in true_m:
                if m_id in targets_by_id and pos_count < 100000:
                    cand = targets_by_id[m_id]
                    if use_expanded_features:
                        f = extract_expanded_pair_features(s1, cand, blocker_score=15.0, blocker_rank=1, name_idf_dict=name_idf)
                    else:
                        f = extract_pair_features(s1, cand)
                    X_rows.append(f)
                    y_rows.append(1)
                    pos_count += 1

            # Hard Negatives
            if hasattr(blocker, "block_entity_with_scores"):
                cands_with_meta = blocker.block_entity_with_scores(s1, top_k=20)
                for cand, b_score, b_rank in cands_with_meta:
                    c_id = cand["entity_id"]
                    if c_id not in true_m and neg_count < 100000:
                        if use_expanded_features:
                            f = extract_expanded_pair_features(s1, cand, blocker_score=b_score, blocker_rank=b_rank, name_idf_dict=name_idf)
                        else:
                            f = extract_pair_features(s1, cand)
                        X_rows.append(f)
                        y_rows.append(0)
                        neg_count += 1
            else:
                for cand in blocker.block_entity(s1, top_k=20):
                    c_id = cand["entity_id"]
                    if c_id not in true_m and neg_count < 100000:
                        f = extract_pair_features(s1, cand)
                        X_rows.append(f)
                        y_rows.append(0)
                        neg_count += 1

            # Additional Hard Negatives (Case A: same name prefix/diff address, Case B: same digit/diff name)
            if mine_hard_negatives and neg_count < 100000:
                for cand in blocker.block_entity(s1, top_k=30)[15:]:
                    c_id = cand["entity_id"]
                    if c_id not in true_m and neg_count < 100000:
                        if use_expanded_features:
                            f = extract_expanded_pair_features(s1, cand, blocker_score=5.0, blocker_rank=20, name_idf_dict=name_idf)
                        else:
                            f = extract_pair_features(s1, cand)
                        X_rows.append(f)
                        y_rows.append(0)
                        neg_count += 1

            if pos_count >= 100000 and neg_count >= 100000:
                break

        if X_rows:
            X_train_list.append(np.array(X_rows, dtype=np.float32))
            y_train_list.append(np.array(y_rows, dtype=np.int32))

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    print(f"Train size: {len(X_train):,d} pairs (Pos: {sum(y_train==1):,d}, Neg: {sum(y_train==0):,d})", flush=True)

    # 3. Train LightGBM
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

    # 4. Evaluate on Validation Set
    val_gt = {s1["entity_id"]: gt_map.get(s1["entity_id"], set()) for s1 in val_s1_subset}
    cand_map: Dict[str, Set[str]] = {}
    preds_map: Dict[str, Set[str]] = {}
    cands_per_s1_list = []

    for s1 in val_s1_subset:
        s1_id = s1["entity_id"]
        cntry = s1["country_norm"].upper()
        blocker = blockers.get(cntry)
        if not blocker:
            cand_map[s1_id] = set()
            preds_map[s1_id] = set()
            cands_per_s1_list.append(0)
            continue

        name_idf = getattr(blocker, "name_idf", None)

        if hasattr(blocker, "block_entity_with_scores"):
            cands_with_meta = blocker.block_entity_with_scores(s1, top_k=top_k)
            cands = [c[0] for c in cands_with_meta]
            b_scores = [c[1] for c in cands_with_meta]
            b_ranks = [c[2] for c in cands_with_meta]
        else:
            cands = blocker.block_entity(s1, top_k=top_k)
            b_scores = [10.0] * len(cands)
            b_ranks = list(range(1, len(cands) + 1))

        cand_ids = [c["entity_id"] for c in cands]
        cand_map[s1_id] = set(cand_ids)
        cands_per_s1_list.append(len(cand_ids))

        if not cands:
            preds_map[s1_id] = set()
            continue

        if use_expanded_features:
            feat_mat = np.array(
                [
                    extract_expanded_pair_features(
                        s1, c, blocker_score=b_scores[i], blocker_rank=b_ranks[i], name_idf_dict=name_idf
                    )
                    for i, c in enumerate(cands)
                ],
                dtype=np.float32,
            )
        else:
            feat_mat = np.array([extract_pair_features(s1, c) for c in cands], dtype=np.float32)

        probs = clf.predict_proba(feat_mat)[:, 1]

        # S1 Decision logic
        if np.max(probs) < tau_singleton:
            preds_map[s1_id] = set()
        else:
            preds_map[s1_id] = {cand_ids[i] for i, p in enumerate(probs) if p >= tau_match}

    report = evaluate(val_gt, preds_map, candidate_pairs=cand_map)
    print_evaluation_report(report)

    total_val_tp = sum(len(m) for m in val_gt.values())
    captured_val_tp = sum(len(val_gt[s1_id].intersection(cand_map[s1_id])) for s1_id in val_gt)
    cand_recall = captured_val_tp / total_val_tp if total_val_tp > 0 else 1.0

    total_fp = sum(len(preds_map[s1_id] - val_gt[s1_id]) for s1_id in val_gt)
    total_fn = sum(len(val_gt[s1_id] - preds_map[s1_id]) for s1_id in val_gt)
    runtime_s = round(time.time() - t_start, 2)

    result = {
        "experiment_id": exp_id,
        "experiment_name": exp_name,
        "blocker_type": blocker_type,
        "top_k": top_k,
        "num_features": len(EXPANDED_FEATURE_COLUMNS if use_expanded_features else FEATURE_COLUMNS),
        "macro_f0.5": report["macro_f0.5"],
        "precision": report["macro_precision"],
        "recall": report["macro_recall"],
        "singleton_accuracy": report["singleton_accuracy"],
        "candidate_recall": round(cand_recall, 6),
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "avg_cands_per_s1": round(float(np.mean(cands_per_s1_list)), 2),
        "median_cands_per_s1": float(np.median(cands_per_s1_list)),
        "max_cands_per_s1": int(np.max(cands_per_s1_list)),
        "tau_singleton": tau_singleton,
        "tau_match": tau_match,
        "runtime_seconds": runtime_s,
    }

    return result, clf, blockers


def run_all_experiments():
    print("=" * 70)
    print("      AMAZON ML CHALLENGE 2026: EXPERIMENT MATRIX RUNNER      ")
    print("=" * 70)

    # 1. Load Data once for strict parity across all experiments
    s1_by_country, targets_by_country, gt_map, all_train_s1, val_s1_subset = load_data_and_split(
        train_dir="dataset/train", max_s1_train=100000, val_size=15000, random_state=42
    )

    experiments = [
        # Exp A: Baseline
        ("Exp_A", "Baseline (Hard Prune, K=25, 23 Feats)", "baseline", 25, False, False, 0.80, 0.70),
        # Exp B: Soft-IDF K=25
        ("Exp_B", "Soft-IDF (K=25, 23 Feats)", "soft_idf", 25, False, False, 0.80, 0.70),
        # Exp C: Soft-IDF K=35
        ("Exp_C", "Soft-IDF (K=35, 23 Feats)", "soft_idf", 35, False, False, 0.80, 0.70),
        # Exp D: Soft-IDF + Char3 K=25
        ("Exp_D", "Soft-IDF + Char3 (K=25, 23 Feats)", "soft_idf_char3", 25, False, False, 0.80, 0.70),
        # Exp E: Soft-IDF + Char3 K=35
        ("Exp_E", "Soft-IDF + Char3 (K=35, 23 Feats)", "soft_idf_char3", 35, False, False, 0.80, 0.70),
        # Exp F: Soft-IDF + Char3 K=35 + Expanded 30 Feats
        ("Exp_F", "Soft-IDF + Char3 (K=35, 30 Feats)", "soft_idf_char3", 35, True, False, 0.80, 0.70),
        # Exp G: Soft-IDF + Char3 K=35 + Expanded 30 Feats + Hard Negatives
        ("Exp_G", "Soft-IDF + Char3 + 30 Feats + Hard Negatives", "soft_idf_char3", 35, True, True, 0.80, 0.70),
    ]

    all_results = []
    best_f05 = -1.0
    best_config = None
    best_model = None

    for exp_id, exp_name, b_type, k_val, use_exp_f, hard_negs, tau_s, tau_m in experiments:
        res, clf, blockers = execute_experiment(
            exp_id=exp_id,
            exp_name=exp_name,
            blocker_type=b_type,
            top_k=k_val,
            use_expanded_features=use_exp_f,
            mine_hard_negatives=hard_negs,
            tau_singleton=tau_s,
            tau_match=tau_m,
            s1_by_country=s1_by_country,
            targets_by_country=targets_by_country,
            gt_map=gt_map,
            all_train_s1=all_train_s1,
            val_s1_subset=val_s1_subset,
        )
        all_results.append(res)

        if res["macro_f0.5"] > best_f05:
            best_f05 = res["macro_f0.5"]
            best_config = res
            best_model = clf

    # Save to CSV
    csv_path = "experiments/results.csv"
    keys = list(all_results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n[SAVED] Full experiment results saved to: {csv_path}")

    # Display Summary Table
    df_res = pd.DataFrame(all_results)
    print("\n" + "=" * 95)
    print("                       EXPERIMENT MATRIX SUMMARY TABLE")
    print("=" * 95)
    print(df_res[["experiment_id", "blocker_type", "top_k", "num_features", "candidate_recall", "macro_f0.5", "precision", "recall", "singleton_accuracy", "runtime_seconds"]].to_string(index=False))
    print("=" * 95)

    return all_results, best_config


if __name__ == "__main__":
    run_all_experiments()
