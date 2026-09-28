"""
Experiment: Multi-Channel Inverted Index with IDF Soft-Weighting vs Hard Pruning.
Tests whether replacing hard token deletion with continuous IDF weighting restores 90%+ recall on full target dataset.
"""

import os
import sys
import time
import math
import re
from collections import defaultdict
from array import array
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.abspath("."))
from src.preprocessing import clean_text_fast, extract_tokens, extract_digits

LEGAL_WORDS = {
    "pvt", "ltd", "private", "limited", "inc", "incorporated", "corp", "corporation",
    "co", "company", "llc", "llp", "plc", "and", "the", "center", "centre", "group",
    "services", "enterprise", "enterprises", "solutions", "tech", "technology",
    "technologies", "associates", "international", "india", "usa"
}


def run_idf_blocking_test(country: str = "US", s1_sample_size: int = 5000):
    print(f"\n=======================================================", flush=True)
    print(f"TESTING IDF-WEIGHTED SOFT BLOCKING FOR: {country}", flush=True)
    print(f"=======================================================", flush=True)

    # 1. Load S1 Sample
    s1_sample = []
    s1_id_set = set()
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                eid, name, addr, cntry = parts[0], parts[1], parts[2], parts[3].strip().upper()
                if cntry == country:
                    s1_sample.append((eid, name, addr))
                    s1_id_set.add(eid)
                    if len(s1_sample) >= s1_sample_size:
                        break

    # 2. Load Ground Truth
    gt_map: Dict[str, Set[str]] = {}
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            eid = parts[0]
            if eid in s1_id_set:
                matches = set(m.strip() for m in parts[1].split(",") if m.strip()) if len(parts) > 1 and parts[1].strip() else set()
                gt_map[eid] = matches

    total_true_matches = sum(len(m) for m in gt_map.values())
    print(f"Loaded {len(s1_sample):,d} S1 records. True matches to capture: {total_true_matches:,d}", flush=True)

    # 3. Index Targets with Soft IDF
    t0 = time.time()
    name_idx = defaultdict(lambda: array("i"))
    prefix_idx = defaultdict(lambda: array("i"))
    digit_idx = defaultdict(lambda: array("i"))
    addr_token_idx = defaultdict(lambda: array("i"))
    target_id_list = []

    for target_file in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        with open(target_file, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    eid, name, addr, cntry = parts[0], parts[1], parts[2], parts[3].strip().upper()
                    if cntry == country:
                        t_int = len(target_id_list)
                        target_id_list.append(eid)

                        # Name tokens
                        for tok in extract_tokens(name):
                            if len(tok) >= 2 and tok not in LEGAL_WORDS:
                                name_idx[tok].append(t_int)

                        # Name prefix
                        clean_n = clean_text_fast(name).replace(" ", "")
                        if len(clean_n) >= 4:
                            prefix_idx[clean_n[:6]].append(t_int)

                        # Digits
                        for dig in extract_digits(addr):
                            digit_idx[dig].append(t_int)

                        # Addr tokens
                        for atok in extract_tokens(addr):
                            if len(atok) >= 4:
                                addr_token_idx[atok].append(t_int)

    N_targets = len(target_id_list)
    print(f"Indexed {N_targets:,d} targets in {time.time() - t0:.2f}s", flush=True)

    # Calculate token IDF weights: log(N / (df + 1))
    name_idf = {k: math.log(N_targets / (len(v) + 1.0)) for k, v in name_idx.items()}
    prefix_idf = {k: math.log(N_targets / (len(v) + 1.0)) for k, v in prefix_idx.items()}
    digit_idf = {k: math.log(N_targets / (len(v) + 1.0)) for k, v in digit_idx.items()}
    addr_idf = {k: math.log(N_targets / (len(v) + 1.0)) for k, v in addr_token_idx.items()}

    # Cap list length for top 0.05% ultra-frequent stop keys rather than deleting
    # Only prune tokens with df > 50,000 (~0.8% of corpus)
    for idx_dict in [name_idx, prefix_idx, digit_idx, addr_token_idx]:
        keys_to_del = [k for k, v in idx_dict.items() if len(v) > 50000]
        for k in keys_to_del:
            del idx_dict[k]

    for K in [25, 35, 50]:
        t_eval = time.time()
        captured = 0
        total_cands = 0

        for s1_id, name, addr in s1_sample:
            true_matches = gt_map.get(s1_id, set())
            scores = defaultdict(float)

            # Name tokens with IDF weight
            for tok in extract_tokens(name):
                if tok in name_idx:
                    idf = name_idf.get(tok, 1.0)
                    w = min(max(idf, 1.0), 12.0)
                    for tid in name_idx[tok]:
                        scores[tid] += w * 1.5

            # Name prefix
            clean_n = clean_text_fast(name).replace(" ", "")
            if len(clean_n) >= 4:
                pref = clean_n[:6]
                if pref in prefix_idx:
                    idf = prefix_idf.get(pref, 1.0)
                    w = min(max(idf, 1.0), 12.0)
                    for tid in prefix_idx[pref]:
                        scores[tid] += w * 1.8

            # Digits
            for dig in extract_digits(addr):
                if dig in digit_idx:
                    idf = digit_idf.get(dig, 1.0)
                    w = min(max(idf, 1.0), 10.0)
                    for tid in digit_idx[dig]:
                        scores[tid] += w * 1.2

            # Addr tokens
            for atok in extract_tokens(addr):
                if atok in addr_token_idx:
                    idf = addr_idf.get(atok, 1.0)
                    w = min(max(idf, 1.0), 8.0)
                    for tid in addr_token_idx[atok]:
                        scores[tid] += w * 0.8

            if not scores:
                continue

            if len(scores) > K:
                top_tids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:K]
            else:
                top_tids = list(scores.keys())

            cand_set = {target_id_list[t] for t in top_tids}
            total_cands += len(cand_set)
            captured += len(true_matches.intersection(cand_set))

        recall = (captured / total_true_matches) * 100.0 if total_true_matches > 0 else 100.0
        avg_c = total_cands / len(s1_sample)
        elapsed = time.time() - t_eval
        print(f"  [Soft IDF Blocker] K={K:2d} | Candidate Recall: {recall:6.2f}% | Avg Cands/S1: {avg_c:5.2f} | Time: {elapsed:5.2f}s", flush=True)


if __name__ == "__main__":
    run_idf_blocking_test("US", s1_sample_size=5000)
