"""
High-Speed Vectorized Test Inference & S1 Decision Engine for Amazon ML Challenge 2026.
Features:
- Country-partitioned execution (FRANCE, INDIA, US).
- 7-Channel High-Recall MultiChannelUnionBlocker.
- 41 High-Dimensional Pairwise Feature Extraction.
- Source-Specific S1-Level Decision Logic (tau_singleton, tau_s2, tau_s3).
- Strictly formatted competition TSV outputs:
  * output/candidate_pairs_optimized.tsv (source1_entity_id\tcandidate_entity_ids)
  * output/matching_results_optimized.tsv (source1_entity_id\tmatched_entity_ids)
"""

import os
import sys
import time
import pickle
import argparse
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.preprocessing import preprocess_record
from src.blocking import MultiChannelUnionBlocker
from src.features import extract_expanded_pair_features, EXPANDED_FEATURE_COLUMNS, extract_pair_features, FEATURE_COLUMNS


def build_country_target_store(
    s2_path: str,
    s3_path: str,
    country: str,
    top_k: int = 35,
    max_token_freq: int = 4000,
) -> MultiChannelUnionBlocker:
    """
    Loads and indexes target records for a country into MultiChannelUnionBlocker.
    """
    t0 = time.time()
    target_records = []

    for target_path in [s2_path, s3_path]:
        with open(target_path, "r", encoding="utf-8") as tf:
            tf.readline()
            for line in tf:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    eid, name, addr, cntry = parts[0], parts[1], parts[2], parts[3].strip().upper()
                    if cntry == country:
                        rec = preprocess_record(eid, name, addr, country)
                        target_records.append(rec)

    blocker = MultiChannelUnionBlocker(default_top_k=top_k, max_token_frequency=max_token_freq, enable_char_ngrams=True)
    blocker.fit_records(target_records)

    print(f"[{country}] Loaded and indexed {len(target_records):,d} target records in {time.time() - t0:.2f}s", flush=True)
    return blocker


def run_country_inference_batch(
    country: str,
    s1_records: List[Dict[str, Any]],
    blocker: MultiChannelUnionBlocker,
    clf: Any,
    tau_singleton: float,
    tau_s2: float,
    tau_s3: float,
    top_k: int,
    cand_writer: Any,
    match_writer: Any,
) -> Tuple[int, int]:
    """
    Batched vectorized inference over preprocessed S1 records using 41 features and source-specific thresholds.
    """
    t0 = time.time()
    total_s1 = len(s1_records)
    total_matches = 0
    batch_size = 2000

    print(f"[{country}] Starting batched inference on {total_s1:,d} S1 records...", flush=True)

    for b_start in range(0, total_s1, batch_size):
        b_end = min(b_start + batch_size, total_s1)
        batch = s1_records[b_start:b_end]

        batch_cand_eids = []
        batch_cand_sources = []
        batch_feat_rows = []
        batch_slice_offsets = []  # [(s1_id, start_idx, end_idx)]

        for s1 in batch:
            s1_id = s1["entity_id"]
            cands_with_scores = blocker.block_entity_with_scores(s1, top_k=top_k)

            if not cands_with_scores:
                batch_slice_offsets.append((s1_id, -1, -1))
                continue

            feat_start = len(batch_feat_rows)
            for cand, score, rank in cands_with_scores:
                c_id = cand["entity_id"]
                src = "S2" if ("S2" in c_id or "s2" in c_id.lower()) else "S3"
                if getattr(clf, "n_features_in_", 41) == len(FEATURE_COLUMNS):
                    batch_feat_rows.append(extract_pair_features(s1, cand))
                else:
                    batch_feat_rows.append(extract_expanded_pair_features(s1, cand, blocker_rank=rank, blocker_score=score))
                batch_cand_eids.append(c_id)
                batch_cand_sources.append(src)
            feat_end = len(batch_feat_rows)

            batch_slice_offsets.append((s1_id, feat_start, feat_end))

        # Batch scoring
        if batch_feat_rows:
            feat_mat = np.array(batch_feat_rows, dtype=np.float32)
            batch_probs = clf.predict_proba(feat_mat)[:, 1]
        else:
            batch_probs = np.array([], dtype=np.float32)

        # Write results for this batch with source-aware thresholds
        for s1_id, start_idx, end_idx in batch_slice_offsets:
            if start_idx == -1:
                cand_writer.write(f"{s1_id}\t\n")
                match_writer.write(f"{s1_id}\t\n")
            else:
                c_ids = batch_cand_eids[start_idx:end_idx]
                c_srcs = batch_cand_sources[start_idx:end_idx]
                p_vals = batch_probs[start_idx:end_idx]

                # 1. Candidate TSV
                cand_writer.write(f"{s1_id}\t{','.join(c_ids)}\n")

                # 2. Matching TSV with source-aware decision
                max_p = np.max(p_vals) if len(p_vals) > 0 else 0.0
                if max_p < tau_singleton:
                    match_writer.write(f"{s1_id}\t\n")
                else:
                    matched = []
                    for i, p in enumerate(p_vals):
                        src = c_srcs[i]
                        thresh = tau_s2 if src == "S2" else tau_s3
                        if p >= thresh:
                            matched.append(c_ids[i])

                    total_matches += len(matched)
                    match_writer.write(f"{s1_id}\t{','.join(matched)}\n")

        if b_end % 50000 == 0 or b_end == total_s1:
            elapsed = time.time() - t0
            speed = b_end / elapsed if elapsed > 0 else 0
            print(f"  [{country}] Processed {b_end:,d} / {total_s1:,d} ({speed:.1f} S1/sec)", flush=True)

    print(f"[{country}] Completed {total_s1:,d} entities in {time.time() - t0:.2f}s | Matches: {total_matches:,d}", flush=True)
    return total_s1, total_matches


def generate_test_predictions(
    dataset_dir: str = "dataset",
    model_path: str = "experiments/exp_union_expanded/entity_resolution_model.pkl",
    output_dir: str = "output",
    output_matching_file: str = "matching_results.tsv",
    output_candidate_file: str = "candidate_pairs.tsv",
    top_k_candidates: int = 35,
    tau_singleton: float = 0.92,
    tau_s2: float = 0.90,
    tau_s3: float = 0.82,
    **kwargs,
) -> Tuple[str, str]:
    """
    Main inference controller for the complete test set.
    """
    test_dir = os.path.join(dataset_dir, "test")
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    # 1. Load Model
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    clf = artifact["model"]
    feat_cols = artifact.get("feature_columns", EXPANDED_FEATURE_COLUMNS)

    os.makedirs(output_dir, exist_ok=True)
    cand_path = os.path.join(output_dir, output_candidate_file)
    match_path = os.path.join(output_dir, output_matching_file)

    print("\n" + "=" * 75)
    print("      AMAZON ML CHALLENGE 2026: TEST INFERENCE ENGINE      ")
    print("=" * 75)
    print(f"MODEL FILE           : {model_path}")
    print(f"BLOCKER ENGINE       : MultiChannelUnionBlocker (7-Channel Union)")
    print(f"TOP-K CANDIDATES     : {top_k_candidates}")
    print(f"FEATURES COUNT       : {len(feat_cols)}")
    print(f"SINGLETON THRESHOLD  : {tau_singleton:.2f} (if max(prob) < {tau_singleton:.2f} -> empty match)")
    print(f"S2 MATCH THRESHOLD   : {tau_s2:.2f} (if candidate from S2 and prob >= {tau_s2:.2f} -> match)")
    print(f"S3 MATCH THRESHOLD   : {tau_s3:.2f} (if candidate from S3 and prob >= {tau_s3:.2f} -> match)")
    print(f"MATCHING TSV OUTPUT  : {match_path}")
    print(f"CANDIDATE TSV OUTPUT : {cand_path}")
    print("=" * 75 + "\n")

    # 2. Read Test S1 records grouped by country
    print(f"[1/3] Reading Test Source 1 records...", flush=True)
    s1_by_country: Dict[str, List[Dict[str, Any]]] = {}
    total_test_s1 = 0

    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                eid, name, addr, cntry = parts[0], parts[1], parts[2], parts[3].strip().upper()
                rec = preprocess_record(eid, name, addr, cntry)
                s1_by_country.setdefault(cntry, []).append(rec)
                total_test_s1 += 1

    print(f"  Total Test S1 Entities: {total_test_s1:,d}")
    for c, records in s1_by_country.items():
        print(f"    - {c:10s}: {len(records):,d} entities")

    # 3. Stream Output Country by Country
    print(f"\n[2/3] Streaming Predictions Country-by-Country...", flush=True)
    with open(cand_path, "w", encoding="utf-8") as cand_f, open(match_path, "w", encoding="utf-8") as match_f:
        cand_f.write("source1_entity_id\tcandidate_entity_ids\n")
        match_f.write("source1_entity_id\tmatched_entity_ids\n")

        total_processed_s1 = 0
        total_predicted_matches = 0

        # Process FRANCE first, then INDIA, then US
        ordered_countries = sorted(s1_by_country.keys(), key=lambda c: len(s1_by_country[c]))

        for country in ordered_countries:
            country_s1_list = s1_by_country[country]
            print(f"\n--- Processing Partition: {country} ({len(country_s1_list):,d} S1 entities) ---", flush=True)

            blocker = build_country_target_store(
                s2_path, s3_path, country, top_k=top_k_candidates, max_token_freq=4000
            )

            n_s1, n_matches = run_country_inference_batch(
                country=country,
                s1_records=country_s1_list,
                blocker=blocker,
                clf=clf,
                tau_singleton=tau_singleton,
                tau_s2=tau_s2,
                tau_s3=tau_s3,
                top_k=top_k_candidates,
                cand_writer=cand_f,
                match_writer=match_f,
            )

            total_processed_s1 += n_s1
            total_predicted_matches += n_matches
            del blocker

    print("\n" + "=" * 75)
    print(f"[3/3] INFERENCE COMPLETE:")
    print(f"  Total S1 Processed : {total_processed_s1:,d} (Expected: {total_test_s1:,d})")
    print(f"  Total Matches Found: {total_predicted_matches:,d}")
    print(f"  Candidate TSV      : {cand_path}")
    print(f"  Matching TSV       : {match_path}")
    print("=" * 75 + "\n")

    return match_path, cand_path


generate_predictions = generate_test_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run High-Speed Batched Test Inference")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--model_path", type=str, default="experiments/exp_union_expanded/entity_resolution_model.pkl")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--output_matching", type=str, default="matching_results_optimized.tsv")
    parser.add_argument("--output_candidate", type=str, default="candidate_pairs_optimized.tsv")
    parser.add_argument("--top_k", type=int, default=35)
    parser.add_argument("--tau_singleton", type=float, default=0.92)
    parser.add_argument("--tau_s2", type=float, default=0.90)
    parser.add_argument("--tau_s3", type=float, default=0.82)
    args = parser.parse_args()

    generate_test_predictions(
        dataset_dir=args.dataset_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        output_matching_file=args.output_matching,
        output_candidate_file=args.output_candidate,
        top_k_candidates=args.top_k,
        tau_singleton=args.tau_singleton,
        tau_s2=args.tau_s2,
        tau_s3=args.tau_s3,
    )
