"""
Benchmark Candidate Recall for MultiChannelUnionBlocker vs Baseline.
Measures candidate recall across Top-K = 25, 35, 50 on the 14,999 Validation Split.
"""

import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Any
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.preprocessing import preprocess_record
from src.blocking import MultiChannelUnionBlocker, CountrySoftIDFBlocker, CountryBlocker, benchmark_blocking_recall


def benchmark_blockers():
    print("=" * 80)
    print("         BENCHMARKING MULTI-CHANNEL UNION BLOCKER vs BASELINE           ")
    print("=" * 80)
    t0 = time.time()

    dataset_dir = "dataset/train"
    s1_path = os.path.join(dataset_dir, "train_source1.tsv")
    s2_path = os.path.join(dataset_dir, "train_source2.tsv")
    s3_path = os.path.join(dataset_dir, "train_source3.tsv")
    gt_path = os.path.join(dataset_dir, "train_ground_truth.tsv")

    # 1. Load 14,999 Validation S1 records
    print("[1/3] Loading 14,999 Validation S1 records...", flush=True)
    val_s1_by_country: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    val_s1_id_set: Set[str] = set()

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
                    val_s1_by_country[cntry].append(rec)
                    val_s1_id_set.add(eid)
                if count >= 114999:
                    break

    # 2. Load Ground Truth
    print("[2/3] Loading Ground Truth...", flush=True)
    gt_map: Dict[str, Set[str]] = {}
    needed_targets: Set[str] = set()

    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            eid = parts[0]
            if eid in val_s1_id_set:
                matches = set(m.strip() for m in parts[1].split(",") if m.strip()) if len(parts) > 1 and parts[1].strip() else set()
                gt_map[eid] = matches
                needed_targets.update(matches)

    total_tp = sum(len(m) for m in gt_map.values())
    print(f"Validation set: {len(val_s1_id_set):,d} S1 entities, {total_tp:,d} true target matches.", flush=True)

    # 3. Load Target Pool
    print("[3/3] Loading 250k Target Records...", flush=True)
    targets_by_country: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    country_bg = {c: 0 for c in val_s1_by_country.keys()}

    for target_path in [s2_path, s3_path]:
        with open(target_path, "r", encoding="utf-8") as tf:
            tf.readline()
            for line in tf:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    eid, name, addr, cntry = parts[0], parts[1], parts[2], parts[3].strip().upper()
                    if cntry in val_s1_by_country:
                        is_needed = eid in needed_targets
                        if is_needed or country_bg[cntry] < 100000:
                            rec = preprocess_record(eid, name, addr, cntry)
                            targets_by_country[cntry].append(rec)
                            if not is_needed:
                                country_bg[cntry] += 1

    configs = [
        ("Baseline CountryBlocker (K=25)", CountryBlocker, 25, {}),
        ("Soft-IDF Blocker (K=25)", CountrySoftIDFBlocker, 25, {"enable_char3": False}),
        ("Multi-Channel Union Blocker (K=25)", MultiChannelUnionBlocker, 25, {"enable_char_ngrams": True}),
        ("Multi-Channel Union Blocker (K=35)", MultiChannelUnionBlocker, 35, {"enable_char_ngrams": True}),
        ("Multi-Channel Union Blocker (K=50)", MultiChannelUnionBlocker, 50, {"enable_char_ngrams": True}),
    ]

    print("\n" + "=" * 80)
    print(f"{'Blocker Configuration':<40} | {'Candidate Recall':<18} | {'Captured TP':<15}")
    print("-" * 80)

    for name, blk_cls, top_k, kwargs in configs:
        captured = 0
        for cntry, s1_list in val_s1_by_country.items():
            t_list = targets_by_country[cntry]
            if blk_cls == CountryBlocker:
                blk = blk_cls(default_top_k=top_k)
            else:
                blk = blk_cls(default_top_k=top_k, **kwargs)
            blk.fit_records(t_list)

            for s1 in s1_list:
                s1_id = s1["entity_id"]
                true_m = gt_map.get(s1_id, set())
                cand_ids = set(blk.block_entity_ids(s1, top_k=top_k))
                captured += len(true_m.intersection(cand_ids))

        rec = captured / total_tp if total_tp > 0 else 1.0
        print(f"{name:<40} | {rec:<18.4%} | {captured:>7,d} / {total_tp:,d}")

    print("=" * 80)
    print(f"Total Benchmark Time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    benchmark_blockers()
