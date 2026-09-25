"""
10-Point Automated Data Audit & Noise Profiler for Amazon ML Challenge 2026.
Runs exploratory data analysis across Source 1, Source 2, Source 3, and Ground Truth.
Profiles record volumes, missing values, open-set country distribution, match cardinality,
and true-positive noise characteristics (Levenshtein, Jaccard, token overlap).
"""

import os
import argparse
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np


def compute_levenshtein_ratio(s1: str, s2: str) -> float:
    """Computes basic Levenshtein similarity ratio between two strings."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    # Optimized 2-row DP for distance
    prev_row = list(range(len2 + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1] * (len2 + 1)
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (0 if c1 == c2 else 1)
            curr_row[j + 1] = min(insertions, deletions, substitutions)
        prev_row = curr_row
    dist = prev_row[len2]
    return 1.0 - (dist / max(len1, len2))


def compute_jaccard_similarity(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Computes Jaccard similarity between two token sets."""
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union > 0 else 0.0


def audit_dataset(dataset_dir: str) -> None:
    """Executes the complete 10-point audit on the dataset directory."""
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    print("\n" + "=" * 75)
    print("      AMAZON ML CHALLENGE 2026: 10-POINT DATA AUDIT & NOISE PROFILING      ")
    print("=" * 75)

    # 1. Record Volume & Integrity
    print("\n[POINT 1 & 2] Record Volumes, Missing Values & Schema Integrity")
    print("-" * 75)
    sources = {}
    for name, path in [("Source 1 (Reference)", s1_path), ("Source 2 (Noisy)", s2_path), ("Source 3 (Noisy)", s3_path)]:
        if not os.path.exists(path):
            print(f"  [MISSING] File not found: {path}")
            continue
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        sources[name] = df
        
        # Check nulls/empties
        null_names = sum(df["business_name"].str.strip() == "")
        null_addrs = sum(df["business_address"].str.strip() == "")
        null_cntry = sum(df["country"].str.strip() == "")
        unique_ids = df["entity_id"].nunique()

        print(f"  * {name:22s}: {len(df):>7,d} rows | Unique IDs: {unique_ids:>7,d}")
        print(f"    - Missing Names     : {null_names:>6,d} ({null_names/len(df)*100:4.1f}%)")
        print(f"    - Missing Addresses : {null_addrs:>6,d} ({null_addrs/len(df)*100:4.1f}%)")
        print(f"    - Missing Countries : {null_cntry:>6,d} ({null_cntry/len(df)*100:4.1f}%)")

    # 3. Country Topology
    print("\n[POINT 3] Country Frequency Distribution (Open-Set Audit)")
    print("-" * 75)
    for name, df in sources.items():
        cntry_counts = df["country"].str.strip().str.upper().value_counts()
        top_str = ", ".join([f"{k}: {v:,}" for k, v in cntry_counts.head(5).items()])
        print(f"  * {name:22s}: {top_str}")

    # Check test countries if test available
    test_s1_path = os.path.join(test_dir, "test_source1.tsv")
    if os.path.exists(test_s1_path):
        test_df = pd.read_csv(test_s1_path, sep="\t", dtype=str, keep_default_na=False)
        test_cntry = test_df["country"].str.strip().str.upper().value_counts()
        print(f"  * Test Source 1 Countries: {', '.join([f'{k}: {v:,}' for k, v in test_cntry.items()])}")

    if not os.path.exists(gt_path):
        print(f"\n[INFO] Ground truth file not found at: {gt_path}. Skipping GT analysis.")
        return

    # 4 & 5. Ground Truth Cardinality & Match Proportions
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    cols = gt_df.columns.tolist()
    s1_col = "source1_entity_id" if "source1_entity_id" in cols else cols[0]
    match_col = "matched_entity_ids" if "matched_entity_ids" in cols else cols[1]

    match_counts = []
    s2_matches = 0
    s3_matches = 0

    s1_dict = {row["entity_id"]: row for _, row in sources.get("Source 1 (Reference)", pd.DataFrame()).iterrows()}
    s2_dict = {row["entity_id"]: row for _, row in sources.get("Source 2 (Noisy)", pd.DataFrame()).iterrows()}
    s3_dict = {row["entity_id"]: row for _, row in sources.get("Source 3 (Noisy)", pd.DataFrame()).iterrows()}

    true_pairs = []

    for _, row in gt_df.iterrows():
        s1_id = str(row[s1_col]).strip()
        matched_str = str(row[match_col]).strip()
        if matched_str and matched_str not in ("nan", "None"):
            matches = [m.strip() for m in matched_str.split(",") if m.strip()]
        else:
            matches = []
        match_counts.append(len(matches))
        for m in matches:
            if m.startswith("S2"):
                s2_matches += 1
                if s1_id in s1_dict and m in s2_dict:
                    true_pairs.append((s1_dict[s1_id], s2_dict[m]))
            elif m.startswith("S3"):
                s3_matches += 1
                if s1_id in s1_dict and m in s3_dict:
                    true_pairs.append((s1_dict[s1_id], s3_dict[m]))

    counts_arr = np.array(match_counts)
    total_s1 = len(match_counts)
    c0 = np.sum(counts_arr == 0)
    c1 = np.sum(counts_arr == 1)
    c2 = np.sum(counts_arr == 2)
    c3_5 = np.sum((counts_arr >= 3) & (counts_arr <= 5))
    c6_plus = np.sum(counts_arr >= 6)

    print("\n[POINT 4 & 5] Ground Truth Cardinality & Match Proportions")
    print("-" * 75)
    print(f"  * Total Ground Truth Entities: {total_s1:,}")
    print(f"  * Singletons (0 Matches)     : {c0:>6,d} ({c0/total_s1*100:5.1f}%) -> Critical for F0.5!")
    print(f"  * Exactly 1 Match            : {c1:>6,d} ({c1/total_s1*100:5.1f}%)")
    print(f"  * Exactly 2 Matches          : {c2:>6,d} ({c2/total_s1*100:5.1f}%)")
    print(f"  * 3 to 5 Matches             : {c3_5:>6,d} ({c3_5/total_s1*100:5.1f}%)")
    print(f"  * 6+ Matches                 : {c6_plus:>6,d} ({c6_plus/total_s1*100:5.1f}%)")
    print(f"  * Mean matches/S1            : {np.mean(counts_arr):.2f} | Median: {np.median(counts_arr):.1f} | Max: {np.max(counts_arr)}")
    total_matches = s2_matches + s3_matches
    if total_matches > 0:
        print(f"  * Matches from Source 2      : {s2_matches:,} ({s2_matches/total_matches*100:.1f}%)")
        print(f"  * Matches from Source 3      : {s3_matches:,} ({s3_matches/total_matches*100:.1f}%)")

    # 6, 7 & 8. Text Length, Duplicates & Numeric Analysis
    print("\n[POINT 6, 7 & 8] Text Length, Duplicates & Numeric Distribution")
    print("-" * 75)
    for name, df in sources.items():
        name_lens = df["business_name"].str.len()
        addr_lens = df["business_address"].str.len()
        has_digits = df["business_address"].str.contains(r"\d", regex=True).mean() * 100
        dup_names = df["business_name"].str.lower().duplicated().mean() * 100
        print(f"  * {name:22s}: Avg Name Len: {name_lens.mean():.1f} ch | Avg Addr Len: {addr_lens.mean():.1f} ch")
        print(f"    - Addresses with Digits: {has_digits:.1f}% | Name Duplicate Rate: {dup_names:.1f}%")

    # 9 & 10. True Positive Similarity & Hard Positive Profiling
    if true_pairs:
        print("\n[POINT 9 & 10] True Positive Similarity & Hard Match Profiling (Sample Size: min(5000, N))")
        print("-" * 75)
        sample_pairs = true_pairs[:5000]
        name_sims = []
        addr_sims = []
        hard_name_pairs = []
        hard_addr_pairs = []

        for r1, r2 in sample_pairs:
            n1 = str(r1.get("business_name", "")).strip().lower()
            n2 = str(r2.get("business_name", "")).strip().lower()
            a1 = str(r1.get("business_address", "")).strip().lower()
            a2 = str(r2.get("business_address", "")).strip().lower()

            ns = compute_levenshtein_ratio(n1, n2)
            as_ = compute_jaccard_similarity(set(a1.split()), set(a2.split()))
            name_sims.append(ns)
            addr_sims.append(as_)

            if ns < 0.50 and as_ > 0.60:
                hard_name_pairs.append((r1, r2, ns, as_))
            elif as_ < 0.20 and ns > 0.80:
                hard_addr_pairs.append((r1, r2, ns, as_))

        print(f"  * Mean True-Pair Name Levenshtein : {np.mean(name_sims):.3f} (Median: {np.median(name_sims):.3f})")
        print(f"  * Mean True-Pair Address Jaccard   : {np.mean(addr_sims):.3f} (Median: {np.median(addr_sims):.3f})")
        
        if hard_name_pairs:
            print(f"\n  [SAMPLE HARD NAME MATCH] (Heavily corrupted name, preserved address):")
            h1, h2, ns, as_ = hard_name_pairs[0]
            print(f"    S1: '{h1['business_name']}' | Addr: '{h1['business_address']}'")
            print(f"    S2/3: '{h2['business_name']}' | Addr: '{h2['business_address']}'")
            print(f"    Name Sim: {ns:.2f}, Addr Sim: {as_:.2f}")

        if hard_addr_pairs:
            print(f"\n  [SAMPLE HARD ADDRESS MATCH] (Preserved name, heavily corrupted address):")
            h1, h2, ns, as_ = hard_addr_pairs[0]
            print(f"    S1: '{h1['business_name']}' | Addr: '{h1['business_address']}'")
            print(f"    S2/3: '{h2['business_name']}' | Addr: '{h2['business_address']}'")
            print(f"    Name Sim: {ns:.2f}, Addr Sim: {as_:.2f}")

    print("\n" + "=" * 75)
    print("      DATA AUDIT COMPLETE — READY FOR ITERATIVE DEVELOPMENT      ")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 10-Point EDA on Amazon Entity Resolution Dataset")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Path to dataset root folder")
    args = parser.parse_args()

    audit_dataset(args.dataset_dir)
