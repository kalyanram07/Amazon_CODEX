"""
Training & Hyperparameter / Threshold Optimization Pipeline for Amazon ML Challenge 2026.
Trains LightGBM pairwise match scorer and tunes S1-level dual thresholds (tau_singleton, tau_match)
to maximize Macro F0.5 on held-out validation entities.
"""

import os
import sys
import time
import pickle
import argparse
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath("."))

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import GradientBoostingClassifier

from src.preprocessing import preprocess_record
from src.blocking import CountryBlocker
from src.features import extract_pair_features, FEATURE_COLUMNS
from src.evaluate import evaluate, print_evaluation_report


def load_training_data(
    train_dir: str,
    max_s1: int = 100000,
    background_targets_per_country: int = 300000,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]], Dict[str, Set[str]]]:
    """
    Efficiently loads S1 records, Ground Truth, and all required + background target records.
    """
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    print(f"Loading top {max_s1:,d} Source 1 records...", flush=True)
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
                if count_s1 >= max_s1:
                    break

    print(f"Loaded {count_s1:,d} Source 1 records. Countries: {list(s1_by_country.keys())}", flush=True)

    print("Loading Ground Truth for selected S1 records...", flush=True)
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

    total_tp = sum(len(m) for m in gt_map.values())
    print(f"Ground truth has {total_tp:,d} true target matches across {len(gt_map):,d} S1 entities.", flush=True)

    print("Loading target records (needed matches + background hard negative pool)...", flush=True)
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

    total_targets_loaded = sum(len(v) for v in targets_by_country.values())
    print(f"Loaded {total_targets_loaded:,d} target records across countries: { {c: len(v) for c, v in targets_by_country.items()} }", flush=True)

    return s1_by_country, targets_by_country, gt_map


def build_training_dataset(
    s1_train_records: List[Dict[str, Any]],
    targets_by_id: Dict[str, Dict[str, Any]],
    gt_map: Dict[str, Set[str]],
    blocker: CountryBlocker,
    max_positives: int = 100000,
    max_negatives: int = 100000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds balanced pairwise training matrix X, y.
    """
    X_rows = []
    y_list = []
    pos_count = 0
    neg_count = 0

    print(f"Mining training pairs from {len(s1_train_records):,d} S1 entities...", flush=True)
    for s1 in s1_train_records:
        s1_id = s1["entity_id"]
        true_matches = gt_map.get(s1_id, set())

        # 1. Positives
        for m_id in true_matches:
            if m_id in targets_by_id and pos_count < max_positives:
                cand = targets_by_id[m_id]
                feats = extract_pair_features(s1, cand)
                X_rows.append(feats)
                y_list.append(1)
                pos_count += 1

        # 2. Hard Negatives from blocker
        cands = blocker.block_entity(s1, top_k=20)
        for cand in cands:
            c_id = cand["entity_id"]
            if c_id not in true_matches and neg_count < max_negatives:
                feats = extract_pair_features(s1, cand)
                X_rows.append(feats)
                y_list.append(0)
                neg_count += 1

        if pos_count >= max_positives and neg_count >= max_negatives:
            break

    print(f"Constructed {len(X_rows):,d} pairs | Positives: {pos_count:,d} | Hard Negatives: {neg_count:,d}", flush=True)
    return np.array(X_rows, dtype=np.float32), np.array(y_list, dtype=np.int32)


def tune_thresholds_on_validation(
    val_s1_records: List[Dict[str, Any]],
    val_blockers: Dict[str, CountryBlocker],
    clf: Any,
    gt_map: Dict[str, Set[str]],
) -> Tuple[float, float, float]:
    """
    Grid searches tau_singleton and tau_match on validation S1 entities to maximize Macro F0.5.
    """
    print(f"\nEvaluating candidate predictions on {len(val_s1_records):,d} validation entities...", flush=True)
    val_gt = {s1["entity_id"]: gt_map.get(s1["entity_id"], set()) for s1 in val_s1_records}
    s1_candidate_data = []

    t0 = time.time()
    for s1 in val_s1_records:
        s1_id = s1["entity_id"]
        cntry = s1["country_norm"].upper()
        blocker = val_blockers.get(cntry)
        if not blocker:
            s1_candidate_data.append((s1_id, [], []))
            continue
        cands = blocker.block_entity(s1, top_k=25)
        if not cands:
            s1_candidate_data.append((s1_id, [], []))
            continue

        cand_ids = [c["entity_id"] for c in cands]
        feat_mat = np.array([extract_pair_features(s1, c) for c in cands], dtype=np.float32)
        probs = clf.predict_proba(feat_mat)[:, 1]
        s1_candidate_data.append((s1_id, cand_ids, probs))

    print(f"Validation feature extraction & inference completed in {time.time() - t0:.2f}s", flush=True)

    best_f05 = -1.0
    best_tau_singleton = 0.70
    best_tau_match = 0.65

    singleton_grid = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    match_grid = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    for tau_s in singleton_grid:
        for tau_m in match_grid:
            preds = {}
            for s1_id, cand_ids, probs in s1_candidate_data:
                if not cand_ids or len(probs) == 0:
                    preds[s1_id] = set()
                    continue
                max_p = np.max(probs)
                if max_p < tau_s:
                    preds[s1_id] = set()
                else:
                    matched = {cand_ids[i] for i, p in enumerate(probs) if p >= tau_m}
                    preds[s1_id] = matched

            report = evaluate(val_gt, preds)
            f05 = report["macro_f0.5"]
            if f05 > best_f05:
                best_f05 = f05
                best_tau_singleton = tau_s
                best_tau_match = tau_m

    print(f"\nOptimal Thresholds Found:")
    print(f"  tau_singleton = {best_tau_singleton:.2f}")
    print(f"  tau_match     = {best_tau_match:.2f}")
    print(f"  Validation Macro F0.5 = {best_f05:.6f}")

    best_preds = {}
    best_cands = {}
    for s1_id, cand_ids, probs in s1_candidate_data:
        best_cands[s1_id] = set(cand_ids)
        if not cand_ids or len(probs) == 0 or np.max(probs) < best_tau_singleton:
            best_preds[s1_id] = set()
        else:
            best_preds[s1_id] = {cand_ids[i] for i, p in enumerate(probs) if p >= best_tau_match}

    best_report = evaluate(val_gt, best_preds, candidate_pairs=best_cands)
    print_evaluation_report(best_report)

    return best_tau_singleton, best_tau_match, best_f05


def train_pipeline(
    dataset_dir: str = "dataset",
    output_model_dir: str = "models",
    max_s1_train: int = 100000,
    val_size: int = 15000,
    random_state: int = 42,
) -> None:
    """End-to-end training and threshold calibration loop."""
    train_dir = os.path.join(dataset_dir, "train")

    print("\n" + "=" * 65)
    print("      AMAZON ML CHALLENGE 2026: MODEL TRAINING & TUNING      ")
    print("=" * 65)

    # 1. Load Data
    s1_by_country, targets_by_country, gt_map = load_training_data(
        train_dir, max_s1=max_s1_train + val_size
    )

    # 2. Train / Val Split grouped by S1 entities across countries
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

    print(f"\nGrouped Split: {len(all_train_s1):,d} Train S1 | {len(all_val_s1):,d} Validation S1")

    # 3. Fit blockers and build training pairs
    X_train_list = []
    y_train_list = []
    val_blockers = {}

    for country in s1_by_country.keys():
        print(f"\nFitting Country Blocker for: {country}...", flush=True)
        country_targets = targets_by_country.get(country, [])
        blocker = CountryBlocker(max_token_frequency=2500, default_top_k=25).fit_records(country_targets)
        val_blockers[country] = blocker

        targets_by_id = {t["entity_id"]: t for t in country_targets}
        country_train_s1 = [s for s in all_train_s1 if s["country_norm"] == country.lower()]

        X_c, y_c = build_training_dataset(
            country_train_s1, targets_by_id, gt_map, blocker,
            max_positives=100000, max_negatives=100000
        )
        X_train_list.append(X_c)
        y_train_list.append(y_c)

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    print(f"\nTotal Training Dataset: {len(X_train):,d} pairs (Positives: {sum(y_train==1):,d}, Hard Negatives: {sum(y_train==0):,d})")

    # 4. Train LightGBM Classifier
    print("\nTraining LightGBM GBDT Classifier...", flush=True)
    t0 = time.time()
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
    print(f"Model trained in {time.time() - t0:.2f}s")

    # Print Feature Importances
    importances = clf.feature_importances_
    print("\nTop Feature Importances:")
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx[:10]:
        print(f"  - {FEATURE_COLUMNS[i]:25s}: {importances[i]}")

    # 5. Threshold Tuning on Validation S1 entities
    print("\n" + "-" * 65)
    print("      THRESHOLD TUNING & LOCAL VALIDATION SCORE      ")
    print("-" * 65)

    best_tau_s, best_tau_m, best_f05 = tune_thresholds_on_validation(
        all_val_s1[:val_size],
        val_blockers,
        clf,
        gt_map,
    )

    # 6. Save Model Artifact
    os.makedirs(output_model_dir, exist_ok=True)
    artifact_path = os.path.join(output_model_dir, "entity_resolution_lgb.pkl")
    model_artifact = {
        "model": clf,
        "feature_columns": FEATURE_COLUMNS,
        "tau_singleton": best_tau_s,
        "tau_match": best_tau_m,
        "validation_macro_f05": best_f05,
    }
    with open(artifact_path, "wb") as f:
        pickle.dump(model_artifact, f)
    print(f"\n[SAVED] Model artifact saved to: {artifact_path}")

    # 7. Update experiment log
    log_path = "experiments/experiment_log.csv"
    log_entry = f"LightGBM_GBDT,{len(X_train)},{best_tau_s:.2f},{best_tau_m:.2f},{best_f05:.6f},{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("model_name,num_pairs,tau_singleton,tau_match,val_macro_f05,timestamp\n")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry)
    print(f"[LOGGED] Recorded run in {log_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM Entity Resolution Model")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--output_model_dir", type=str, default="models")
    parser.add_argument("--max_s1_train", type=int, default=100000)
    parser.add_argument("--val_size", type=int, default=15000)
    args = parser.parse_args()

    train_pipeline(
        dataset_dir=args.dataset_dir,
        output_model_dir=args.output_model_dir,
        max_s1_train=args.max_s1_train,
        val_size=args.val_size,
    )
