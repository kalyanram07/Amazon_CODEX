"""
Stage 7: S1 Margin & Cluster Ranking Decision Engine.
Evaluates margin-based dynamic acceptance:
- Margin = p1 - p2
- Dual-candidate cluster detection (p1 >= tau_s and p2 >= tau_cluster)
- High-margin single acceptance (p1 >= tau_single and margin >= tau_margin)
- Ambiguity suppression (rejecting borderline candidates with tiny margin)
"""

import os
import sys
import time
import json
import pickle
import numpy as np
from typing import Dict, List, Set, Tuple, Any

sys.path.insert(0, os.path.abspath("."))

from src.evaluate import evaluate, print_evaluation_report


def evaluate_margin_decision_engine():
    print("=" * 80)
    print("         STAGE 7: S1 MARGIN & CLUSTER RANKING DECISION ENGINE          ")
    print("=" * 80)

    # Load experiment model & validation results
    model_path = "experiments/exp_union_expanded/entity_resolution_model.pkl"
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    clf = artifact["model"]

    # We will run inference on the validation S1 entities
    # To be fast, let's load the cached validation predictions if available or run quick inference
    from src.preprocessing import preprocess_record
    from src.blocking import MultiChannelUnionBlocker
    from src.features import extract_expanded_pair_features, EXPANDED_FEATURE_COLUMNS

    dataset_dir = "dataset/train"
    s1_path = os.path.join(dataset_dir, "train_source1.tsv")
    s2_path = os.path.join(dataset_dir, "train_source2.tsv")
    s3_path = os.path.join(dataset_dir, "train_source3.tsv")
    gt_path = os.path.join(dataset_dir, "train_ground_truth.tsv")

    val_s1_records = []
    val_s1_id_set = set()
    val_s1_by_country = {"INDIA": [], "US": []}

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
                    val_s1_id_set.add(eid)
                    if cntry in val_s1_by_country:
                        val_s1_by_country[cntry].append(rec)
                if count >= 114999:
                    break

    gt_map = {}
    needed_targets = set()
    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            eid = parts[0]
            if eid in val_s1_id_set:
                matches = set(m.strip() for m in parts[1].split(",") if m.strip()) if len(parts) > 1 and parts[1].strip() else set()
                gt_map[eid] = matches
                needed_targets.update(matches)

    targets_by_country = {"INDIA": [], "US": []}
    country_bg_count = {"INDIA": 0, "US": 0}
    for target_path in [s2_path, s3_path]:
        with open(target_path, "r", encoding="utf-8") as tf:
            tf.readline()
            for line in tf:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    eid, name, addr, cntry = parts[0], parts[1], parts[2], parts[3].strip().upper()
                    if cntry in targets_by_country:
                        is_needed = eid in needed_targets
                        if is_needed or country_bg_count[cntry] < 125000:
                            rec = preprocess_record(eid, name, addr, cntry)
                            targets_by_country[cntry].append(rec)
                            if not is_needed:
                                country_bg_count[cntry] += 1

    blockers = {}
    for cntry, t_list in targets_by_country.items():
        blk = MultiChannelUnionBlocker(default_top_k=35, max_token_frequency=4000, enable_char_ngrams=True)
        blk.fit_records(t_list)
        blockers[cntry] = blk

    print("Running validation inference for margin grid search...", flush=True)
    s1_eval_data = []
    for s1 in val_s1_records:
        s1_id = s1["entity_id"]
        cntry = s1.get("country_norm", "").upper()
        blk = blockers.get(cntry)
        if not blk:
            s1_eval_data.append((s1_id, [], []))
            continue

        cands_with_scores = blk.block_entity_with_scores(s1, top_k=35)
        if not cands_with_scores:
            s1_eval_data.append((s1_id, [], []))
            continue

        cand_ids = [c[0]["entity_id"] for c in cands_with_scores]
        feat_mat = np.array([
            extract_expanded_pair_features(s1, c[0], blocker_rank=c[2], blocker_score=c[1])
            for c in cands_with_scores
        ], dtype=np.float32)

        probs = clf.predict_proba(feat_mat)[:, 1]
        s1_eval_data.append((s1_id, cand_ids, probs))

    # Grid search over margin and cluster rules
    best_f05 = -1.0
    best_params = {}

    tau_s_list = [0.85, 0.88, 0.90, 0.92, 0.94]
    tau_m_list = [0.75, 0.80, 0.82, 0.85, 0.88]
    min_margin_list = [0.0, 0.05, 0.10, 0.15, 0.20]

    for tau_s in tau_s_list:
        for tau_m in tau_m_list:
            for min_m in min_margin_list:
                preds = {}
                for s1_id, c_ids, probs in s1_eval_data:
                    if not c_ids or len(probs) == 0:
                        preds[s1_id] = set()
                        continue

                    # Sort candidates by probability
                    sorted_indices = np.argsort(-probs)
                    p1 = probs[sorted_indices[0]]
                    p2 = probs[sorted_indices[1]] if len(sorted_indices) > 1 else 0.0
                    margin = p1 - p2

                    if p1 < tau_s:
                        preds[s1_id] = set()
                    elif margin < min_m and p1 < 0.95:
                        # Ambiguous low-margin candidate
                        preds[s1_id] = set()
                    else:
                        matched = {c_ids[i] for i, p in enumerate(probs) if p >= tau_m}
                        preds[s1_id] = matched

                report = evaluate(gt_map, preds)
                f05 = report["macro_f0.5"]
                if f05 > best_f05:
                    best_f05 = f05
                    best_params = {
                        "tau_singleton": tau_s,
                        "tau_match": tau_m,
                        "min_margin": min_m,
                        "macro_f05": f05,
                        "macro_precision": report["macro_precision"],
                        "macro_recall": report["macro_recall"],
                        "singleton_accuracy": report["singleton_accuracy"],
                    }

    print("\n" + "=" * 80)
    print("                OPTIMAL S1 MARGIN & RANKING PARAMETERS                 ")
    print("=" * 80)
    for k, v in best_params.items():
        if isinstance(v, float):
            print(f"  {k:<25}: {v:.6f}")
        else:
            print(f"  {k:<25}: {v}")
    print("=" * 80)

    # Save to experiments/best_margin_engine/
    out_dir = "experiments/best_margin_engine"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "margin_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)


if __name__ == "__main__":
    evaluate_margin_decision_engine()
