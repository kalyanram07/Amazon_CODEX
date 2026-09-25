"""
Diagnostic Error Analysis Module for Amazon ML Challenge 2026.
Classifies validation errors into:
1. False Merges (False Positives) -> Identifies features causing false matches
2. Missed Matches (False Negatives) -> Separates blocking misses from threshold rejections
3. Singleton Contaminations -> False matches added to true singletons
"""

import argparse
from typing import Dict, List, Set, Tuple
import pandas as pd
from src.evaluate import parse_ground_truth, parse_predictions


def analyze_errors(
    gt_path: str,
    pred_path: str,
    cand_path: str = None,
    s1_path: str = None,
    s2_path: str = None,
    s3_path: str = None,
    max_examples: int = 5,
) -> None:
    """Performs deep error diagnostics on predictions."""
    gt_map = parse_ground_truth(gt_path)
    pred_map = parse_predictions(pred_path)
    cand_map = parse_predictions(cand_path) if cand_path else {}

    s1_dict, s2_dict, s3_dict = {}, {}, {}
    if s1_path:
        df1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
        s1_dict = {r["entity_id"]: r.to_dict() for _, r in df1.iterrows()}
    if s2_path:
        df2 = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False)
        s2_dict = {r["entity_id"]: r.to_dict() for _, r in df2.iterrows()}
    if s3_path:
        df3 = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False)
        s3_dict = {r["entity_id"]: r.to_dict() for _, r in df3.iterrows()}

    all_target_lookup = {**s2_dict, **s3_dict}

    false_merges = []
    blocking_misses = []
    threshold_misses = []
    singleton_errors = []

    for s1_id, true_set in gt_map.items():
        pred_set = pred_map.get(s1_id, set())
        cand_set = cand_map.get(s1_id, set())

        # Singleton errors
        if len(true_set) == 0 and len(pred_set) > 0:
            singleton_errors.append((s1_id, pred_set))

        # False Merges (FP)
        fps = pred_set - true_set
        if fps:
            false_merges.append((s1_id, true_set, fps))

        # Missed Matches (FN)
        fns = true_set - pred_set
        for fn in fns:
            if fn not in cand_set and cand_map:
                blocking_misses.append((s1_id, fn))
            else:
                threshold_misses.append((s1_id, fn))

    print("\n" + "=" * 70)
    print("           DIAGNOSTIC ERROR ANALYSIS REPORT           ")
    print("=" * 70)
    print(f"  Total S1 Evaluated        : {len(gt_map):,d}")
    print(f"  False Merge Entities (FP) : {len(false_merges):,d} (Destroys F0.5 Precision!)")
    print(f"  Singleton Contaminations  : {len(singleton_errors):,d}")
    if cand_map:
        print(f"  Missed at Blocking Stage  : {len(blocking_misses):,d} (Recall Ceiling Deficit)")
        print(f"  Missed at Threshold Stage : {len(threshold_misses):,d} (Confidence < Threshold)")
    print("-" * 70)

    # Sample False Merges
    if false_merges and s1_dict:
        print("\n[SAMPLE FALSE MERGES (Investigate why model was overconfident)]:")
        for s1_id, trues, fps in false_merges[:max_examples]:
            s1_info = s1_dict.get(s1_id, {})
            print(f"  * S1 ({s1_id}): '{s1_info.get('business_name', '')}' | '{s1_info.get('business_address', '')}'")
            for fp in fps:
                fp_info = all_target_lookup.get(fp, {})
                print(f"    - INCORRECT MERGE ({fp}): '{fp_info.get('business_name', '')}' | '{fp_info.get('business_address', '')}'")

    # Sample Blocking Misses
    if blocking_misses and s1_dict:
        print("\n[SAMPLE BLOCKING MISSES (Investigate why blocker did not index)]:")
        for s1_id, fn in blocking_misses[:max_examples]:
            s1_info = s1_dict.get(s1_id, {})
            fn_info = all_target_lookup.get(fn, {})
            print(f"  * S1 ({s1_id}): '{s1_info.get('business_name', '')}' | '{s1_info.get('business_address', '')}'")
            print(f"    - MISSED TRUE MATCH ({fn}): '{fn_info.get('business_name', '')}' | '{fn_info.get('business_address', '')}'")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Entity Resolution Prediction Errors")
    parser.add_argument("--gt", type=str, required=True, help="Ground truth TSV")
    parser.add_argument("--pred", type=str, required=True, help="Prediction TSV")
    parser.add_argument("--cand", type=str, default=None, help="Optional Candidate TSV")
    parser.add_argument("--s1", type=str, default=None)
    parser.add_argument("--s2", type=str, default=None)
    parser.add_argument("--s3", type=str, default=None)
    args = parser.parse_args()

    analyze_errors(
        args.gt, args.pred, cand_path=args.cand, s1_path=args.s1, s2_path=args.s2, s3_path=args.s3
    )
