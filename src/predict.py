"""
High-Speed Vectorized Test Inference & S1 Decision Engine for Amazon ML Challenge 2026.
Features:
- Country-partitioned execution (FRANCE, INDIA, US).
- Single-pass pre-parsing of tokens and normalized strings (0 duplicate work).
- Batch-vectorized LightGBM probability scoring (1000 S1 records per batch).
- Strictly formatted competition TSV outputs:
  * output/candidate_pairs.tsv (source1_entity_id\tcandidate_entity_ids)
  * output/matching_results.tsv (source1_entity_id\tmatched_entity_ids)
"""

import os
import sys
import time
import pickle
import argparse
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
from array import array
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.preprocessing import preprocess_record
from src.features import extract_pair_features, FEATURE_COLUMNS


LEGAL_WORDS = {
    "pvt", "ltd", "private", "limited", "inc", "incorporated", "corp", "corporation",
    "co", "company", "llc", "llp", "plc", "and", "the", "center", "centre", "group",
    "services", "enterprise", "enterprises", "solutions", "tech", "technology",
    "technologies", "associates", "international", "india", "usa", "france"
}


def build_country_target_store(
    s2_path: str,
    s3_path: str,
    country: str,
    max_token_freq: int = 2500,
) -> Tuple[List[Dict[str, Any]], Dict[str, array], Dict[str, array], Dict[str, array], Dict[str, array]]:
    """
    Loads and indexes target records for a country once.
    """
    t0 = time.time()
    target_records = []
    name_idx = defaultdict(lambda: array("i"))
    prefix_idx = defaultdict(lambda: array("i"))
    digit_idx = defaultdict(lambda: array("i"))
    addr_token_idx = defaultdict(lambda: array("i"))

    idx = 0
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

                        for tok in rec.get("name_tokens", []):
                            if len(tok) >= 2 and tok not in LEGAL_WORDS:
                                name_idx[tok].append(idx)
                        pref = rec.get("name_prefix", "")
                        if pref:
                            prefix_idx[pref].append(idx)
                        for dig in rec.get("addr_digits", []):
                            digit_idx[dig].append(idx)
                        for tok in rec.get("addr_tokens", []):
                            if len(tok) >= 4:
                                addr_token_idx[tok].append(idx)
                        idx += 1

    # Prune stop keys
    for idx_dict in [name_idx, prefix_idx, digit_idx, addr_token_idx]:
        keys_to_del = [k for k, v in idx_dict.items() if len(v) > max_token_freq]
        for k in keys_to_del:
            del idx_dict[k]

    print(f"[{country}] Loaded and indexed {len(target_records):,d} target records in {time.time() - t0:.2f}s", flush=True)
    return target_records, name_idx, prefix_idx, digit_idx, addr_token_idx


def run_country_inference_batch(
    country: str,
    s1_records: List[Dict[str, Any]],
    target_records: List[Dict[str, Any]],
    name_idx: Dict[str, array],
    prefix_idx: Dict[str, array],
    digit_idx: Dict[str, array],
    addr_token_idx: Dict[str, array],
    clf: Any,
    tau_singleton: float,
    tau_match: float,
    top_k: int,
    cand_writer: Any,
    match_writer: Any,
) -> Tuple[int, int]:
    """
    Batched inference over preprocessed S1 records.
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
        batch_feat_rows = []
        batch_slice_offsets = []  # [(s1_id, start_idx, end_idx)]

        for s1 in batch:
            s1_id = s1["entity_id"]
            scores = defaultdict(float)

            for tok in s1.get("name_tokens", []):
                if tok in name_idx:
                    for tid in name_idx[tok]:
                        scores[tid] += 3.0

            pref = s1.get("name_prefix", "")
            if pref and pref in prefix_idx:
                for tid in prefix_idx[pref]:
                    scores[tid] += 4.0

            for dig in s1.get("addr_digits", []):
                if dig in digit_idx:
                    for tid in digit_idx[dig]:
                        scores[tid] += 2.0

            for tok in s1.get("addr_tokens", []):
                if tok in addr_token_idx:
                    for tid in addr_token_idx[tok]:
                        scores[tid] += 1.0

            if not scores:
                batch_slice_offsets.append((s1_id, -1, -1))
                continue

            if len(scores) > top_k:
                top_ints = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
            else:
                top_ints = list(scores.keys())

            c_eids = [target_records[idx]["entity_id"] for idx in top_ints]
            feat_start = len(batch_feat_rows)
            for idx in top_ints:
                c_dict = target_records[idx]
                batch_feat_rows.append(extract_pair_features(s1, c_dict))
                batch_cand_eids.append(c_dict["entity_id"])
            feat_end = len(batch_feat_rows)

            batch_slice_offsets.append((s1_id, feat_start, feat_end))

        # Batch scoring
        if batch_feat_rows:
            feat_mat = np.array(batch_feat_rows, dtype=np.float32)
            batch_probs = clf.predict_proba(feat_mat)[:, 1]
        else:
            batch_probs = np.array([], dtype=np.float32)

        # Write results for this batch
        for s1_id, start_idx, end_idx in batch_slice_offsets:
            if start_idx == -1:
                cand_writer.write(f"{s1_id}\t\n")
                match_writer.write(f"{s1_id}\t\n")
            else:
                c_ids = batch_cand_eids[start_idx:end_idx]
                p_vals = batch_probs[start_idx:end_idx]

                cand_writer.write(f"{s1_id}\t{','.join(c_ids)}\n")

                if np.max(p_vals) < tau_singleton:
                    match_writer.write(f"{s1_id}\t\n")
                else:
                    matched = [c_ids[i] for i, p in enumerate(p_vals) if p >= tau_match]
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
    model_path: str = "models/entity_resolution_lgb.pkl",
    output_dir: str = "output",
    top_k_candidates: int = 25,
) -> Tuple[str, str]:
    """
    Main inference controller for the complete test set.
    """
    test_dir = os.path.join(dataset_dir, "test")
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    print("\n" + "=" * 70)
    print("      AMAZON ML CHALLENGE 2026: ULTRA-FAST TEST INFERENCE      ")
    print("=" * 70)

    # 1. Load Model
    print(f"\n[1/4] Loading Model from: {model_path}...", flush=True)
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    clf = artifact["model"]
    tau_s = artifact.get("tau_singleton", 0.80)
    tau_m = artifact.get("tau_match", 0.70)
    print(f"  Model Loaded. Thresholds: tau_singleton = {tau_s:.2f}, tau_match = {tau_m:.2f}")

    # 2. Read Test S1 records grouped by country
    print(f"\n[2/4] Reading Test Source 1 records...", flush=True)
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

    # 3. Stream Output
    os.makedirs(output_dir, exist_ok=True)
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")

    print(f"\n[3/4] Streaming Predictions Country-by-Country...", flush=True)
    with open(cand_path, "w", encoding="utf-8") as cand_f, open(match_path, "w", encoding="utf-8") as match_f:
        cand_f.write("source1_entity_id\tcandidate_entity_ids\n")
        match_f.write("source1_entity_id\tmatched_entity_ids\n")

        total_processed_s1 = 0
        total_predicted_matches = 0

        # Process FRANCE first, then US, then INDIA
        ordered_countries = sorted(s1_by_country.keys(), key=lambda c: len(s1_by_country[c]))

        for country in ordered_countries:
            country_s1_list = s1_by_country[country]
            print(f"\n--- Processing Partition: {country} ({len(country_s1_list):,d} S1 entities) ---", flush=True)

            target_records, name_idx, prefix_idx, digit_idx, addr_token_idx = build_country_target_store(
                s2_path, s3_path, country, max_token_freq=2500
            )

            n_s1, n_matches = run_country_inference_batch(
                country=country,
                s1_records=country_s1_list,
                target_records=target_records,
                name_idx=name_idx,
                prefix_idx=prefix_idx,
                digit_idx=digit_idx,
                addr_token_idx=addr_token_idx,
                clf=clf,
                tau_singleton=tau_s,
                tau_match=tau_m,
                top_k=top_k_candidates,
                cand_writer=cand_f,
                match_writer=match_f,
            )

            total_processed_s1 += n_s1
            total_predicted_matches += n_matches
            del target_records, name_idx, prefix_idx, digit_idx, addr_token_idx

    print("\n" + "=" * 70)
    print(f"[4/4] INFERENCE COMPLETE:")
    print(f"  Total S1 Processed : {total_processed_s1:,d} (Expected: {total_test_s1:,d})")
    print(f"  Total Matches Found: {total_predicted_matches:,d}")
    print(f"  Candidate TSV      : {cand_path}")
    print(f"  Matching TSV       : {match_path}")
    print("=" * 70 + "\n")

    return match_path, cand_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run High-Speed Batched Test Inference")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--model_path", type=str, default="models/entity_resolution_lgb.pkl")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--top_k", type=int, default=25)
    args = parser.parse_args()

    generate_test_predictions(
        dataset_dir=args.dataset_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        top_k_candidates=args.top_k,
    )
