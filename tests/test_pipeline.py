"""
End-to-End Pipeline Unit & Integration Test.
Creates a realistic mock challenge dataset with singletons, hard positives,
hard negatives, and open-set countries (France), executes preprocessing,
blocking, feature extraction, model training, S1-level inference, and F0.5 evaluation.
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.preprocessing import preprocess_dataframe
from src.blocking import MultiStrategyBlocker, benchmark_blocking_recall
from src.features import PairFeatureExtractor
from src.train import train_model
from src.predict import generate_predictions
from src.evaluate import parse_ground_truth, parse_predictions, evaluate, print_evaluation_report


def setup_mock_dataset(base_dir: str = "mock_data"):
    """Creates a mock dataset covering all edge cases."""
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 1. Source 1 (Reference)
    s1_data = [
        {"entity_id": "S1-00001", "business_name": "Starbucks Coffee", "business_address": "12 MG Road, Bangalore", "country": "India"},
        {"entity_id": "S1-00002", "business_name": "Apple Store", "business_address": "5th Avenue, New York, NY 10022", "country": "US"},
        {"entity_id": "S1-00003", "business_name": "Unique Local Bakery", "business_address": "77 Baker Street, London", "country": "UK"}, # Singleton
        {"entity_id": "S1-00004", "business_name": "Boulangerie Paul", "business_address": "84 Rue de Rivoli, Paris", "country": "France"}, # France test
    ]
    pd.DataFrame(s1_data).to_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", index=False)
    pd.DataFrame(s1_data).to_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t", index=False)

    # 2. Source 2 (Noisy)
    s2_data = [
        {"entity_id": "S2-00101", "business_name": "Starbucks Coffee Pvt Ltd", "business_address": "12 M.G. Rd Bengaluru", "country": "India"}, # Match S1-00001
        {"entity_id": "S2-00102", "business_name": "Starbucks Hyderabad", "business_address": "99 Banjara Hills, Hyderabad", "country": "India"}, # Hard Negative for S1-00001
        {"entity_id": "S2-00103", "business_name": "Apple Inc", "business_address": "5th Ave, NY", "country": "US"}, # Match S1-00002
        {"entity_id": "S2-00104", "business_name": "Paul Boulangerie SAS", "business_address": "84 Rue de Rivoli, 75004 Paris", "country": "France"}, # Match S1-00004
    ]
    pd.DataFrame(s2_data).to_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t", index=False)
    pd.DataFrame(s2_data).to_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t", index=False)

    # 3. Source 3 (Noisy)
    s3_data = [
        {"entity_id": "S3-00201", "business_name": "Starbucks Cafe", "business_address": "Near 12 MG Road Metro, Bengaluru", "country": "India"}, # Match S1-00001
        {"entity_id": "S3-00202", "business_name": "Random Unrelated Corp", "business_address": "100 Main St, Chicago", "country": "US"},
    ]
    pd.DataFrame(s3_data).to_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t", index=False)
    pd.DataFrame(s3_data).to_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t", index=False)

    # 4. Ground Truth
    gt_data = [
        {"source1_entity_id": "S1-00001", "matched_entity_ids": "S2-00101,S3-00201"},
        {"source1_entity_id": "S1-00002", "matched_entity_ids": "S2-00103"},
        {"source1_entity_id": "S1-00003", "matched_entity_ids": ""}, # True singleton
        {"source1_entity_id": "S1-00004", "matched_entity_ids": "S2-00104"}, # France match
    ]
    pd.DataFrame(gt_data).to_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", index=False)
    print("Mock dataset generated successfully.")


def run_pipeline_test():
    """Runs complete end-to-end integration test."""
    mock_dir = "mock_data"
    setup_mock_dataset(mock_dir)

    print("\n--- TEST 1: Training Model ---")
    train_model(dataset_dir=mock_dir, model_type="logistic", val_split_ratio=0.25, output_model_dir="mock_models")

    print("\n--- TEST 2: Running Inference & S1 Decision Engine ---")
    generate_predictions(
        dataset_dir=mock_dir,
        is_test=False,
        model_path="mock_models/entity_resolution_model.pkl",
        tau_singleton=0.50,
        tau_match=0.50,
        output_dir="mock_output",
    )

    print("\n--- TEST 3: Evaluating Macro F0.5 & Invariants ---")
    gt = parse_ground_truth(os.path.join(mock_dir, "train", "train_ground_truth.tsv"))
    preds = parse_predictions("mock_output/matching_results.tsv")
    cands = parse_predictions("mock_output/candidate_pairs.tsv")

    report = evaluate(gt, preds, candidate_pairs=cands)
    print_evaluation_report(report)

    # Invariant assertions
    assert report["subset_violations"] == 0, "Subset invariant violated!"
    assert report["macro_f0.5"] > 0.0, "Macro F0.5 score must be positive!"
    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")

    # Cleanup mock data
    if os.path.exists(mock_dir):
        shutil.rmtree(mock_dir)
    if os.path.exists("mock_models"):
        shutil.rmtree("mock_models")
    if os.path.exists("mock_output"):
        shutil.rmtree("mock_output")


if __name__ == "__main__":
    run_pipeline_test()
