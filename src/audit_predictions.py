"""
Comprehensive post-inference distribution audit and integrity verification script.
"""
import os
import sys
from collections import Counter, defaultdict


def audit_results(
    dataset_dir: str = "dataset",
    output_dir: str = "output",
):
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    test_s1_path = os.path.join(dataset_dir, "test", "test_source1.tsv")

    print("\n" + "=" * 70)
    print("      AMAZON ML CHALLENGE 2026: POST-INFERENCE AUDIT      ")
    print("=" * 70)

    # 1. Load Expected Test S1 entities
    print(f"\n[1/5] Loading Expected Test S1 Entities from {test_s1_path}...")
    s1_countries = {}
    with open(test_s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                eid, country = parts[0], parts[3].strip().upper()
                s1_countries[eid] = country

    total_expected_s1 = len(s1_countries)
    print(f"  Expected S1 Entities: {total_expected_s1:,d}")
    print(f"  Country distribution: {Counter(s1_countries.values())}")

    # 2. Check Candidate Pairs
    print(f"\n[2/5] Inspecting Candidate Pairs from {candidate_path}...")
    candidate_map = {}
    total_candidates = 0
    with open(candidate_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        assert header == ["source1_entity_id", "candidate_entity_ids"], f"Bad candidate header: {header}"
        for line in f:
            parts = line.strip().split("\t")
            eid = parts[0]
            cands = [c.strip() for c in parts[1].split(",") if c.strip()] if len(parts) > 1 and parts[1].strip() else []
            candidate_map[eid] = set(cands)
            total_candidates += len(cands)

    print(f"  Total Candidate Rows: {len(candidate_map):,d}")
    print(f"  Total Candidate Pairs: {total_candidates:,d} (Avg {total_candidates/len(candidate_map):.1f} per S1)")

    # 3. Check Matching Results
    print(f"\n[3/5] Inspecting Matching Results from {matching_path}...")
    matching_map = {}
    match_count_by_s1 = Counter()
    s2_matches = 0
    s3_matches = 0
    matches_by_country = defaultdict(int)
    singletons_by_country = defaultdict(int)

    with open(matching_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        assert header == ["source1_entity_id", "matched_entity_ids"], f"Bad matching header: {header}"
        for line in f:
            parts = line.strip().split("\t")
            eid = parts[0]
            matches = [m.strip() for m in parts[1].split(",") if m.strip()] if len(parts) > 1 and parts[1].strip() else []
            matching_map[eid] = matches
            n_m = len(matches)
            match_count_by_s1[n_m] += 1

            cntry = s1_countries.get(eid, "UNKNOWN")
            matches_by_country[cntry] += n_m
            if n_m == 0:
                singletons_by_country[cntry] += 1

            for m in matches:
                if m.startswith("S2"):
                    s2_matches += 1
                elif m.startswith("S3"):
                    s3_matches += 1

    total_matched_rows = len(matching_map)
    total_matches = sum(len(m) for m in matching_map.values())

    print(f"  Total Matching Rows: {total_matched_rows:,d}")
    print(f"  Total Predicted Matches: {total_matches:,d} (Avg {total_matches/total_matched_rows:.2f} per S1)")
    print(f"  Source-2 Matches: {s2_matches:,d} ({s2_matches/total_matches*100:.1f}%)")
    print(f"  Source-3 Matches: {s3_matches:,d} ({s3_matches/total_matches*100:.1f}%)")

    # 4. Match Distribution & Singletons
    print("\n[4/5] Match Cardinality Breakdown:")
    n_zero = match_count_by_s1[0]
    n_one = match_count_by_s1[1]
    n_multi = sum(v for k, v in match_count_by_s1.items() if k >= 2)
    print(f"  - 0 Matches (Singletons): {n_zero:,d} ({n_zero/total_matched_rows*100:.2f}%)")
    print(f"  - 1 Match               : {n_one:,d} ({n_one/total_matched_rows*100:.2f}%)")
    print(f"  - 2+ Matches            : {n_multi:,d} ({n_multi/total_matched_rows*100:.2f}%)")

    print("\n  Country-Specific Match Breakdown:")
    for cntry, total_cnt in Counter(s1_countries.values()).items():
        n_m = matches_by_country[cntry]
        n_s = singletons_by_country[cntry]
        print(f"    * {cntry:8s}: {total_cnt:,d} S1s | {n_m:,d} matches (avg {n_m/total_cnt:.2f}) | {n_s:,d} singletons ({n_s/total_cnt*100:.1f}%)")

    # 5. Strict Invariant & Integrity Checks
    print("\n[5/5] Invariant & Integrity Verification:")
    errors = []

    # Check 1: Row count equals expected S1 count
    if total_matched_rows != total_expected_s1:
        errors.append(f"Row count mismatch in matching_results: {total_matched_rows} vs expected {total_expected_s1}")
    if len(candidate_map) != total_expected_s1:
        errors.append(f"Row count mismatch in candidate_pairs: {len(candidate_map)} vs expected {total_expected_s1}")

    # Check 2: Matching subset of candidate
    subset_violations = 0
    for eid, matches in matching_map.items():
        cand_set = candidate_map.get(eid, set())
        for m in matches:
            if m not in cand_set:
                subset_violations += 1
                if subset_violations <= 5:
                    errors.append(f"Subset violation for {eid}: match {m} not in candidates")

    if subset_violations > 0:
        errors.append(f"Total subset violations: {subset_violations}")
    else:
        print("  [PASS] Subset Invariant: matching_results is 100% a subset of candidate_pairs.")

    if not errors:
        print("  [PASS] All structural, cardinality, and invariant checks PASSED successfully!")
        return True
    else:
        print(f"  [FAIL] Errors detected: {len(errors)}")
        for err in errors[:10]:
            print(f"    - {err}")
        return False


if __name__ == "__main__":
    success = audit_results()
    sys.exit(0 if success else 1)
