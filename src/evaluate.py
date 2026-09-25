"""
Official Metric Evaluator for Amazon ML Challenge 2026: Business Entity Resolution.
Calculates per-S1 entity F0.5, Precision, Recall, and overall Macro-Averaged F0.5.
Handles singletons, subset verification, and per-entity diagnostic reporting.
"""

import argparse
from typing import Dict, Set, Tuple, Optional
import pandas as pd
import numpy as np


def parse_ground_truth(gt_path: str) -> Dict[str, Set[str]]:
    """
    Parses ground truth TSV into a mapping: {s1_id: set(matched_s2_s3_ids)}.
    Expected format: source1_entity_id\tmatched_entity_ids (or entity_id\tmatched_entity_ids)
    """
    df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    
    # Standardize column names
    cols = df.columns.tolist()
    s1_col = "source1_entity_id" if "source1_entity_id" in cols else ("s1_entity_id" if "s1_entity_id" in cols else cols[0])
    match_col = "matched_entity_ids" if "matched_entity_ids" in cols else cols[1]

    gt_dict = {}
    for _, row in df.iterrows():
        s1_id = str(row[s1_col]).strip()
        matched_str = str(row[match_col]).strip()
        if matched_str and matched_str != "nan" and matched_str != "None":
            matches = set(m.strip() for m in matched_str.split(",") if m.strip())
        else:
            matches = set()
        gt_dict[s1_id] = matches
    return gt_dict


def parse_predictions(pred_path: str) -> Dict[str, Set[str]]:
    """
    Parses prediction TSV into a mapping: {s1_id: set(matched_s2_s3_ids)}.
    """
    df = pd.read_csv(pred_path, sep="\t", dtype=str, keep_default_na=False)
    cols = df.columns.tolist()
    s1_col = "source1_entity_id" if "source1_entity_id" in cols else ("s1_entity_id" if "s1_entity_id" in cols else cols[0])
    match_col = "matched_entity_ids" if "matched_entity_ids" in cols else cols[1]

    pred_dict = {}
    for _, row in df.iterrows():
        s1_id = str(row[s1_col]).strip()
        matched_str = str(row[match_col]).strip()
        if matched_str and matched_str != "nan" and matched_str != "None":
            matches = set(m.strip() for m in matched_str.split(",") if m.strip())
        else:
            matches = set()
        pred_dict[s1_id] = matches
    return pred_dict


def compute_entity_f_beta(
    actual: Set[str], predicted: Set[str], beta: float = 0.5
) -> Tuple[float, float, float]:
    """
    Computes precision, recall, and F-beta for a single S1 entity.
    Exact Singleton Rules:
    - If actual is empty and predicted is empty: P=1.0, R=1.0, F=1.0 (Correct Singleton)
    - If actual is empty and predicted is non-empty: P=0.0, R=1.0, F=0.0 (False Positive Merge)
    - If actual is non-empty and predicted is empty: P=1.0, R=0.0, F=0.0 (Missed Matches)
    """
    if len(actual) == 0 and len(predicted) == 0:
        return 1.0, 1.0, 1.0
    if len(actual) == 0 and len(predicted) > 0:
        return 0.0, 1.0, 0.0
    if len(actual) > 0 and len(predicted) == 0:
        return 1.0, 0.0, 0.0

    tp = len(actual.intersection(predicted))
    fp = len(predicted - actual)
    fn = len(actual - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    beta_sq = beta ** 2
    denominator = (beta_sq * precision) + recall
    if denominator > 0:
        f_beta = (1 + beta_sq) * (precision * recall) / denominator
    else:
        f_beta = 0.0

    return precision, recall, f_beta


def evaluate(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    candidate_pairs: Optional[Dict[str, Set[str]]] = None,
    beta: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluates predictions against ground truth for all S1 entities in ground truth.
    Also validates the subset invariant: Matches(S1) <= Candidates(S1) if candidates are provided.
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        raise ValueError("Ground truth dictionary is empty.")

    f_beta_scores = []
    precision_scores = []
    recall_scores = []

    singletons_count = 0
    correct_singletons = 0
    contaminated_singletons = 0
    subset_violations = 0

    for s1_id, actual_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())

        # Verify candidate subset constraint
        if candidate_pairs is not None:
            cand_set = candidate_pairs.get(s1_id, set())
            if not pred_set.issubset(cand_set):
                subset_violations += 1

        is_singleton = len(actual_set) == 0
        if is_singleton:
            singletons_count += 1
            if len(pred_set) == 0:
                correct_singletons += 1
            else:
                contaminated_singletons += 1

        prec, rec, fb = compute_entity_f_beta(actual_set, pred_set, beta=beta)
        precision_scores.append(prec)
        recall_scores.append(rec)
        f_beta_scores.append(fb)

    macro_f_beta = float(np.mean(f_beta_scores))
    macro_precision = float(np.mean(precision_scores))
    macro_recall = float(np.mean(recall_scores))
    singleton_accuracy = (correct_singletons / singletons_count) if singletons_count > 0 else 1.0

    report = {
        "macro_f0.5": round(macro_f_beta, 6),
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "total_s1_entities": total_entities,
        "singleton_count": singletons_count,
        "correct_singletons": correct_singletons,
        "contaminated_singletons": contaminated_singletons,
        "singleton_accuracy": round(singleton_accuracy, 6),
        "subset_violations": subset_violations,
    }

    return report


def print_evaluation_report(report: Dict[str, float]) -> None:
    """Pretty prints evaluation metrics."""
    print("=" * 60)
    print("      AMAZON ML CHALLENGE 2026: EVALUATION REPORT      ")
    print("=" * 60)
    print(f"  Macro F0.5 Score        : {report['macro_f0.5']:.6f}")
    print(f"  Macro Precision         : {report['macro_precision']:.6f}")
    print(f"  Macro Recall            : {report['macro_recall']:.6f}")
    print("-" * 60)
    print(f"  Evaluated S1 Entities   : {report['total_s1_entities']:,}")
    print(f"  True Singletons         : {report['singleton_count']:,}")
    print(f"  Correct Singletons (1.0): {report['correct_singletons']:,}")
    print(f"  False Merge Singletons  : {report['contaminated_singletons']:,}")
    print(f"  Singleton Accuracy      : {report['singleton_accuracy'] * 100:.2f}%")
    if report["subset_violations"] > 0:
        print(f"  [CRITICAL WARNING] Subset Violations: {report['subset_violations']:,}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Entity Resolution Predictions against Ground Truth")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth TSV")
    parser.add_argument("--pred", type=str, required=True, help="Path to prediction TSV (matching_results.tsv)")
    parser.add_argument("--cand", type=str, default=None, help="Optional: Path to candidate pairs TSV")
    args = parser.parse_args()

    gt = parse_ground_truth(args.gt)
    preds = parse_predictions(args.pred)
    cands = parse_predictions(args.cand) if args.cand else None

    metrics = evaluate(gt, preds, candidate_pairs=cands)
    print_evaluation_report(metrics)
