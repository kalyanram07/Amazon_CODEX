"""
Controlled Optimization Sequence:
Experiment A: Full Model K-Sweep (K=25 vs K=35 vs K=50)
Experiment B: Fine Dual-Threshold Grid Sweep
Experiment C: Source-Specific Calibration (tau_s2 vs tau_s3 vs tau_singleton)
Experiment D: Country Breakdown (US vs India) & Generalization Check
"""

import os
import sys
import time
import json
import pickle
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.preprocessing import preprocess_record
from src.blocking import MultiChannelUnionBlocker
from src.features import extract_expanded_pair_features, EXPANDED_FEATURE_COLUMNS
from src.evaluate import evaluate, print_evaluation_report


def run_controlled_optimization():
    print("=" * 80)
    print("      STAGE 8: CONTROLLED K-SWEEP, FINE THRESHOLDS & SOURCE CALIBRATION      ")
    print("=" * 80)
    t_start = time.time()

    dataset_dir = "dataset/train"
    s1_path = os.path.join(dataset_dir, "train_source1.tsv")
    s2_path = os.path.join(dataset_dir, "train_source2.tsv")
    s3_path = os.path.join(dataset_dir, "train_source3.tsv")
    gt_path = os.path.join(dataset_dir, "train_ground_truth.tsv")

    # 1. Load Validation S1 Records (14,999 entities)
    print("[1/4] Loading 14,999 validation S1 entities...", flush=True)
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

    # 2. Load Ground Truth
    print("[2/4] Loading Ground Truth...", flush=True)
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

    total_tp = sum(len(m) for m in gt_map.values())
    print(f"Validation set: {len(val_s1_id_set):,d} S1 entities, {total_tp:,d} true target matches.", flush=True)

    # 3. Load Target Pool (all needed matches + 250k background records)
    print("[3/4] Loading Target Pool...", flush=True)
    targets_by_country: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    country_bg_count = {"INDIA": 0, "US": 0}

    for target_path in [s2_path, s3_path]:
        with open(target_path, "r", encoding="utf-8") as tf:
            tf.readline()
            for line in tf:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    eid, name, addr, cntry = parts[0], parts[1], parts[2], parts[3].strip().upper()
                    if cntry in val_s1_by_country:
                        is_needed = eid in needed_targets
                        if is_needed or country_bg_count.get(cntry, 0) < 125000:
                            rec = preprocess_record(eid, name, addr, cntry)
                            targets_by_country[cntry].append(rec)
                            if not is_needed and cntry in country_bg_count:
                                country_bg_count[cntry] += 1

    # Load 41-feature model
    print("[4/4] Loading trained 41-feature LightGBM Model artifact...", flush=True)
    model_path = "experiments/exp_union_expanded/entity_resolution_model.pkl"
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    clf = artifact["model"]

    # Fit 7-Channel Union Blockers
    blockers: Dict[str, MultiChannelUnionBlocker] = {}
    for cntry, t_list in targets_by_country.items():
        blk = MultiChannelUnionBlocker(default_top_k=50, max_token_frequency=4000, enable_char_ngrams=True)
        blk.fit_records(t_list)
        blockers[cntry] = blk

    print("\nPre-computing Top-50 candidates & probabilities for all 14,999 validation entities...", flush=True)
    t_inf = time.time()
    s1_all_cand_data = []  # list of (s1_id, country, [(cand_id, source, prob, blocker_score, blocker_rank)])

    for s1 in val_s1_records:
        s1_id = s1["entity_id"]
        cntry = s1.get("country_norm", "").upper()
        blk = blockers.get(cntry)
        if not blk:
            s1_all_cand_data.append((s1_id, cntry, []))
            continue

        cands_with_scores = blk.block_entity_with_scores(s1, top_k=50)
        if not cands_with_scores:
            s1_all_cand_data.append((s1_id, cntry, []))
            continue

        feat_mat = np.array([
            extract_expanded_pair_features(s1, c[0], blocker_rank=c[2], blocker_score=c[1])
            for c in cands_with_scores
        ], dtype=np.float32)

        probs = clf.predict_proba(feat_mat)[:, 1]

        cand_info = []
        for i, (c, score, rank) in enumerate(cands_with_scores):
            cid = c["entity_id"]
            src = "S2" if ("S2" in cid or "s2" in cid.lower()) else "S3"
            cand_info.append((cid, src, float(probs[i]), score, rank))

        s1_all_cand_data.append((s1_id, cntry, cand_info))

    print(f"Pre-computation completed in {time.time() - t_inf:.2f}s for 14,999 entities.", flush=True)

    # =========================================================================
    # EXPERIMENT A: FULL MODEL K-SWEEP (K=25 vs K=35 vs K=50)
    # =========================================================================
    print("\n" + "=" * 80)
    print("               EXPERIMENT A: FULL MODEL K-SWEEP (Macro F0.5)           ")
    print("=" * 80)
    print(f"{'Top-K':<10} | {'Macro F0.5':<15} | {'Precision':<15} | {'Recall':<15} | {'Singleton Acc':<15}")
    print("-" * 80)

    k_results = {}
    for k_val in [25, 35, 50]:
        best_k_f05 = -1.0
        best_k_report = None
        best_k_taus = (0.92, 0.88)

        for tau_s in [0.90, 0.92, 0.94]:
            for tau_m in [0.84, 0.86, 0.88, 0.90]:
                preds = {}
                for s1_id, cntry, cand_info in s1_all_cand_data:
                    k_cands = cand_info[:k_val]
                    if not k_cands:
                        preds[s1_id] = set()
                        continue
                    p_list = [c[2] for c in k_cands]
                    max_p = max(p_list)
                    if max_p < tau_s:
                        preds[s1_id] = set()
                    else:
                        preds[s1_id] = {c[0] for c in k_cands if c[2] >= tau_m}

                report = evaluate(gt_map, preds)
                if report["macro_f0.5"] > best_k_f05:
                    best_k_f05 = report["macro_f0.5"]
                    best_k_report = report
                    best_k_taus = (tau_s, tau_m)

        k_results[k_val] = {
            "macro_f05": best_k_f05,
            "precision": best_k_report["macro_precision"],
            "recall": best_k_report["macro_recall"],
            "singleton_accuracy": best_k_report["singleton_accuracy"],
            "best_tau_singleton": best_k_taus[0],
            "best_tau_match": best_k_taus[1],
        }
        print(f"K = {k_val:<6} | {best_k_f05:<15.6f} | {best_k_report['macro_precision']:<15.4%} | {best_k_report['macro_recall']:<15.4%} | {best_k_report['singleton_accuracy']:<15.4%}")

    # =========================================================================
    # EXPERIMENT B: FINE DUAL THRESHOLD GRID SWEEP (At best K=35)
    # =========================================================================
    print("\n" + "=" * 80)
    print("               EXPERIMENT B: FINE DUAL THRESHOLD GRID SWEEP            ")
    print("=" * 80)

    fine_best_f05 = -1.0
    fine_best_params = {}
    fine_grid_results = []

    for tau_s in [0.88, 0.90, 0.92, 0.94, 0.96]:
        for tau_m in [0.82, 0.84, 0.86, 0.88, 0.90, 0.92]:
            preds = {}
            for s1_id, cntry, cand_info in s1_all_cand_data:
                k_cands = cand_info[:35]
                if not k_cands:
                    preds[s1_id] = set()
                    continue
                p_list = [c[2] for c in k_cands]
                max_p = max(p_list)
                if max_p < tau_s:
                    preds[s1_id] = set()
                else:
                    preds[s1_id] = {c[0] for c in k_cands if c[2] >= tau_m}

            report = evaluate(gt_map, preds)
            f05 = report["macro_f0.5"]
            fine_grid_results.append((tau_s, tau_m, f05, report["macro_precision"], report["macro_recall"]))
            if f05 > fine_best_f05:
                fine_best_f05 = f05
                fine_best_params = {
                    "tau_singleton": tau_s,
                    "tau_match": tau_m,
                    "macro_f05": f05,
                    "macro_precision": report["macro_precision"],
                    "macro_recall": report["macro_recall"],
                    "singleton_accuracy": report["singleton_accuracy"],
                }

    print(f"Optimal tau_singleton : {fine_best_params['tau_singleton']:.2f}")
    print(f"Optimal tau_match     : {fine_best_params['tau_match']:.2f}")
    print(f"Peak Macro F0.5       : {fine_best_f05:.6f} (Prec: {fine_best_params['macro_precision']:.2%}, Rec: {fine_best_params['macro_recall']:.2%})")

    # =========================================================================
    # EXPERIMENT C: SOURCE-SPECIFIC CALIBRATION (tau_s2 != tau_s3)
    # =========================================================================
    print("\n" + "=" * 80)
    print("               EXPERIMENT C: SOURCE-SPECIFIC THRESHOLD CALIBRATION      ")
    print("=" * 80)

    src_best_f05 = -1.0
    src_best_params = {}

    for tau_s in [0.90, 0.92, 0.94]:
        for tau_s2 in [0.84, 0.86, 0.88, 0.90, 0.92]:
            for tau_s3 in [0.80, 0.82, 0.84, 0.86, 0.88]:
                preds = {}
                for s1_id, cntry, cand_info in s1_all_cand_data:
                    k_cands = cand_info[:35]
                    if not k_cands:
                        preds[s1_id] = set()
                        continue
                    p_list = [c[2] for c in k_cands]
                    max_p = max(p_list)
                    if max_p < tau_s:
                        preds[s1_id] = set()
                    else:
                        matched = set()
                        for cid, src, prob, _, _ in k_cands:
                            thresh = tau_s2 if src == "S2" else tau_s3
                            if prob >= thresh:
                                matched.add(cid)
                        preds[s1_id] = matched

                report = evaluate(gt_map, preds)
                f05 = report["macro_f0.5"]
                if f05 > src_best_f05:
                    src_best_f05 = f05
                    src_best_params = {
                        "tau_singleton": tau_s,
                        "tau_s2": tau_s2,
                        "tau_s3": tau_s3,
                        "macro_f05": f05,
                        "macro_precision": report["macro_precision"],
                        "macro_recall": report["macro_recall"],
                        "singleton_accuracy": report["singleton_accuracy"],
                    }

    print(f"Optimal tau_singleton : {src_best_params['tau_singleton']:.2f}")
    print(f"Optimal tau_S2        : {src_best_params['tau_s2']:.2f}")
    print(f"Optimal tau_S3        : {src_best_params['tau_s3']:.2f}")
    print(f"Source-Tuned Macro F0.5: {src_best_f05:.6f} (Prec: {src_best_params['macro_precision']:.2%}, Rec: {src_best_params['macro_recall']:.2%})")

    # =========================================================================
    # EXPERIMENT D: COUNTRY-LEVEL PERFORMANCE BREAKDOWN
    # =========================================================================
    print("\n" + "=" * 80)
    print("               EXPERIMENT D: COUNTRY PERFORMANCE BREAKDOWN             ")
    print("=" * 80)

    # Evaluate per-country with best parameters
    tau_s = src_best_params["tau_singleton"]
    tau_s2 = src_best_params["tau_s2"]
    tau_s3 = src_best_params["tau_s3"]

    preds_by_country = defaultdict(dict)
    gt_by_country = defaultdict(dict)

    for s1_id, cntry, cand_info in s1_all_cand_data:
        gt_by_country[cntry][s1_id] = gt_map.get(s1_id, set())
        k_cands = cand_info[:35]
        if not k_cands or max(c[2] for c in k_cands) < tau_s:
            preds_by_country[cntry][s1_id] = set()
        else:
            matched = set()
            for cid, src, prob, _, _ in k_cands:
                thresh = tau_s2 if src == "S2" else tau_s3
                if prob >= thresh:
                    matched.add(cid)
            preds_by_country[cntry][s1_id] = matched

    print(f"{'Country':<15} | {'Entities':<12} | {'Macro F0.5':<15} | {'Precision':<15} | {'Recall':<15}")
    print("-" * 80)
    for cntry, c_gt in gt_by_country.items():
        c_preds = preds_by_country[cntry]
        rep = evaluate(c_gt, c_preds)
        print(f"{cntry:<15} | {len(c_gt):<12,d} | {rep['macro_f0.5']:<15.6f} | {rep['macro_precision']:<15.4%} | {rep['macro_recall']:<15.4%}")

    # Save summary report
    out_dir = "experiments/final_optimized_config"
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "optimization_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "k_sweep": k_results,
            "fine_grid_best": fine_best_params,
            "source_specific_best": src_best_params,
            "total_elapsed_time": round(time.time() - t_start, 2),
        }, f, indent=2)

    print(f"\n[SAVED] Controlled optimization summary saved to: {summary_path}")
    print(f"Total Optimization Time: {time.time() - t_start:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    run_controlled_optimization()
