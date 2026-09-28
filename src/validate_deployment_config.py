"""
Step 2 Verification: Evaluate Locked Deployment Configuration on 14,999 Validation Entities.
Outputs exact metric breakdown:
- Macro F0.5
- Precision
- Recall
- Singleton Accuracy
- US Macro F0.5
- India Macro F0.5
- S2 Macro F0.5
- S3 Macro F0.5
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


def validate_locked_deployment():
    print("=" * 80)
    print("      STEP 2: FINAL VALIDATION CHECK ON LOCKED DEPLOYMENT CONFIGURATION      ")
    print("=" * 80)
    t0 = time.time()

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

    # 3. Load Target Pool
    print("[3/4] Loading Target Pool (needed matches + 250k background records)...", flush=True)
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

    # 4. Load Model and Fit Blocker
    print("[4/4] Loading Model and evaluating locked deployment configuration...", flush=True)
    model_path = "experiments/exp_union_expanded/entity_resolution_model.pkl"
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    clf = artifact["model"]

    blockers: Dict[str, MultiChannelUnionBlocker] = {}
    for cntry, t_list in targets_by_country.items():
        blk = MultiChannelUnionBlocker(default_top_k=35, max_token_frequency=4000, enable_char_ngrams=True)
        blk.fit_records(t_list)
        blockers[cntry] = blk

    tau_singleton = 0.92
    tau_s2 = 0.90
    tau_s3 = 0.82
    top_k = 35

    preds = {}
    preds_by_country = defaultdict(dict)
    gt_by_country = defaultdict(dict)
    source_results = {"S2": {"tp": 0, "fp": 0, "fn": 0}, "S3": {"tp": 0, "fp": 0, "fn": 0}}
    cands_map = {}

    for s1 in val_s1_records:
        s1_id = s1["entity_id"]
        cntry = s1.get("country_norm", "").upper()
        true_m = gt_map.get(s1_id, set())
        gt_by_country[cntry][s1_id] = true_m

        blk = blockers.get(cntry)
        if not blk:
            preds[s1_id] = set()
            preds_by_country[cntry][s1_id] = set()
            continue

        cands_with_scores = blk.block_entity_with_scores(s1, top_k=top_k)
        if not cands_with_scores:
            preds[s1_id] = set()
            preds_by_country[cntry][s1_id] = set()
            continue

        cand_ids = [c[0]["entity_id"] for c in cands_with_scores]
        cands_map[s1_id] = set(cand_ids)

        feat_mat = np.array([
            extract_expanded_pair_features(s1, c[0], blocker_rank=c[2], blocker_score=c[1])
            for c in cands_with_scores
        ], dtype=np.float32)

        probs = clf.predict_proba(feat_mat)[:, 1]

        max_p = np.max(probs) if len(probs) > 0 else 0.0
        if max_p < tau_singleton:
            pred_set = set()
        else:
            pred_set = set()
            for i, p in enumerate(probs):
                cid = cand_ids[i]
                src = "S2" if ("S2" in cid or "s2" in cid.lower()) else "S3"
                thresh = tau_s2 if src == "S2" else tau_s3
                if p >= thresh:
                    pred_set.add(cid)

        preds[s1_id] = pred_set
        preds_by_country[cntry][s1_id] = pred_set

        # Source breakdown
        for mid in pred_set:
            src = "S2" if ("S2" in mid or "s2" in mid.lower()) else "S3"
            if mid in true_m:
                source_results[src]["tp"] += 1
            else:
                source_results[src]["fp"] += 1
        for mid in (true_m - pred_set):
            src = "S2" if ("S2" in mid or "s2" in mid.lower()) else "S3"
            source_results[src]["fn"] += 1

    # Compute overall report
    final_report = evaluate(gt_map, preds, candidate_pairs=cands_map)

    # Compute country reports
    us_report = evaluate(gt_by_country["US"], preds_by_country["US"])
    india_report = evaluate(gt_by_country["INDIA"], preds_by_country["INDIA"])

    # Compute Source F0.5
    s2_p = source_results["S2"]["tp"] / (source_results["S2"]["tp"] + source_results["S2"]["fp"]) if (source_results["S2"]["tp"] + source_results["S2"]["fp"]) > 0 else 1.0
    s2_r = source_results["S2"]["tp"] / (source_results["S2"]["tp"] + source_results["S2"]["fn"]) if (source_results["S2"]["tp"] + source_results["S2"]["fn"]) > 0 else 1.0
    s2_f05 = 1.25 * (s2_p * s2_r) / (0.25 * s2_p + s2_r) if (s2_p + s2_r) > 0 else 0.0

    s3_p = source_results["S3"]["tp"] / (source_results["S3"]["tp"] + source_results["S3"]["fp"]) if (source_results["S3"]["tp"] + source_results["S3"]["fp"]) > 0 else 1.0
    s3_r = source_results["S3"]["tp"] / (source_results["S3"]["tp"] + source_results["S3"]["fn"]) if (source_results["S3"]["tp"] + source_results["S3"]["fn"]) > 0 else 1.0
    s3_f05 = 1.25 * (s3_p * s3_r) / (0.25 * s3_p + s3_r) if (s3_p + s3_r) > 0 else 0.0

    print("\n" + "=" * 80)
    print("      LOCKED DEPLOYMENT CONFIGURATION FINAL VALIDATION REPORT      ")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Score / Value':<20}")
    print("-" * 80)
    print(f"{'Overall Macro F0.5':<30} | {final_report['macro_f0.5']:<20.6f}")
    print(f"{'Macro Precision':<30} | {final_report['macro_precision']:<20.4%}")
    print(f"{'Macro Recall':<30} | {final_report['macro_recall']:<20.4%}")
    print(f"{'Candidate Blocker Recall':<30} | {final_report.get('candidate_recall', 0.0):<20.4%}")
    print(f"{'Singleton Accuracy':<30} | {final_report['singleton_accuracy']:<20.4%}")
    print(f"{'US Macro F0.5':<30} | {us_report['macro_f0.5']:<20.6f}")
    print(f"{'India Macro F0.5':<30} | {india_report['macro_f0.5']:<20.6f}")
    print(f"{'Source 2 (S2) F0.5':<30} | {s2_f05:<20.6f}")
    print(f"{'Source 3 (S3) F0.5':<30} | {s3_f05:<20.6f}")
    print("=" * 80)
    print(f"Total Validation Time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    validate_locked_deployment()
