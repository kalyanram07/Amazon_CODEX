"""
Comprehensive Benchmark for Blocker Strategies: Top-K Sweep + Character 3-gram Channels.
Measures candidate recall, average candidate count, and runtime on validation ground truth for US and India.
"""

import os
import sys
import time
import re
import string
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


def extract_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extracts compact character n-grams from alphanumeric text."""
    clean = re.sub(r"[^a-z0-9]", "", text.lower())
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i:i+n] for i in range(len(clean) - n + 1)}


def run_blocking_experiment(country: str, s1_sample_size: int = 10000):
    print(f"\n=======================================================", flush=True)
    print(f"RUNNING BLOCKING EXPERIMENT FOR COUNTRY: {country} (S1 Sample: {s1_sample_size:,d})", flush=True)
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

    print(f"Loaded {len(s1_sample):,d} S1 records.", flush=True)

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
    print(f"Total true ground truth matches to capture: {total_true_matches:,d}", flush=True)

    # 3. Index Targets
    t0 = time.time()
    name_idx = defaultdict(lambda: array("i"))
    prefix_idx = defaultdict(lambda: array("i"))
    digit_idx = defaultdict(lambda: array("i"))
    addr_token_idx = defaultdict(lambda: array("i"))
    char3_idx = defaultdict(lambda: array("i"))
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
                        toks = extract_tokens(name)
                        for tok in toks:
                            if len(tok) >= 2 and tok not in LEGAL_WORDS:
                                name_idx[tok].append(t_int)

                        # Name prefix
                        clean_n = clean_text_fast(name).replace(" ", "")
                        if len(clean_n) >= 4:
                            prefix_idx[clean_n[:6]].append(t_int)

                        # Address digits
                        for dig in extract_digits(addr):
                            digit_idx[dig].append(t_int)

                        # Address tokens
                        for atok in extract_tokens(addr):
                            if len(atok) >= 4:
                                addr_token_idx[atok].append(t_int)

                        # Char 3-grams
                        for c3 in extract_char_ngrams(name, n=3):
                            char3_idx[c3].append(t_int)

    # Prune high frequency keys
    for idx_dict, max_f in [(name_idx, 2500), (prefix_idx, 2500), (digit_idx, 2500), (addr_token_idx, 2500), (char3_idx, 1500)]:
        keys_to_del = [k for k, v in idx_dict.items() if len(v) > max_f]
        for k in keys_to_del:
            del idx_dict[k]

    print(f"Indexed {len(target_id_list):,d} targets in {time.time() - t0:.2f}s", flush=True)

    # Evaluate configurations
    configs = [
        ("Baseline (No Char3)", False),
        ("With Char 3-gram Channel", True),
    ]
    k_values = [25, 30, 40, 50]

    for label, use_char3 in configs:
        print(f"\n--- Strategy: {label} ---", flush=True)
        for K in k_values:
            t_eval = time.time()
            captured = 0
            total_cands = 0

            for s1_id, name, addr in s1_sample:
                true_matches = gt_map.get(s1_id, set())
                scores = defaultdict(float)

                # Name tokens
                for tok in extract_tokens(name):
                    if tok in name_idx:
                        for tid in name_idx[tok]:
                            scores[tid] += 3.0

                # Name prefix
                clean_n = clean_text_fast(name).replace(" ", "")
                if len(clean_n) >= 4:
                    pref = clean_n[:6]
                    if pref in prefix_idx:
                        for tid in prefix_idx[pref]:
                            scores[tid] += 4.0

                # Digits
                for dig in extract_digits(addr):
                    if dig in digit_idx:
                        for tid in digit_idx[dig]:
                            scores[tid] += 2.0

                # Addr tokens
                for atok in extract_tokens(addr):
                    if atok in addr_token_idx:
                        for tid in addr_token_idx[atok]:
                            scores[tid] += 1.0

                # Optional Char3
                if use_char3:
                    for c3 in extract_char_ngrams(name, n=3):
                        if c3 in char3_idx:
                            for tid in char3_idx[c3]:
                                scores[tid] += 0.8

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
            print(f"  K={K:2d} | Candidate Recall: {recall:6.2f}% | Avg Cands/S1: {avg_c:5.2f} | Time: {elapsed:5.2f}s", flush=True)


if __name__ == "__main__":
    for c in ["US", "INDIA"]:
        run_blocking_experiment(c, s1_sample_size=10000)
