"""
Ultra-fast, memory-efficient Blocking Recall Benchmark with Country Partitioning & Integer IDs.
"""
import sys
import time
import re
import string
from collections import defaultdict
from array import array

LEGAL_WORDS = {"pvt", "ltd", "private", "limited", "inc", "incorporated", "corp", "corporation", 
               "co", "company", "llc", "llp", "plc", "and", "the", "center", "centre", "group", "services",
               "enterprise", "enterprises", "solutions", "tech", "technologies", "associates", "international"}

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

def get_name_tokens(name: str) -> set:
    if not name:
        return set()
    cleaned = clean_text(name)
    tokens = set()
    words = cleaned.split()
    for w in words:
        if len(w) >= 2 and w not in LEGAL_WORDS:
            tokens.add(w)
    # Compressed name prefix for domain/merged names
    compact = "".join(words)
    if len(compact) >= 4:
        tokens.add(compact[:8])
    return tokens

def get_addr_tokens(addr: str) -> set:
    if not addr:
        return set()
    cleaned = clean_text(addr)
    tokens = set()
    for w in cleaned.split():
        if len(w) >= 4:
            tokens.add(w)
    return tokens

def get_digits(addr: str) -> set:
    if not addr:
        return set()
    return set(re.findall(r"\b\d+\b", addr))

def run_country_benchmark(target_country: str = "US", s1_limit: int = 10000):
    print(f"\n=======================================================", flush=True)
    print(f"BENCHMARKING BLOCKING FOR COUNTRY: {target_country}", flush=True)
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
                if cntry == target_country:
                    s1_sample.append((eid, name, addr))
                    s1_id_set.add(eid)
                    if len(s1_sample) >= s1_limit:
                        break
                        
    print(f"Loaded {len(s1_sample):,d} S1 records for {target_country}", flush=True)
    
    # 2. Load Ground Truth for this sample
    gt_map = {}
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            eid = parts[0]
            if eid in s1_id_set:
                matches = set(m.strip() for m in parts[1].split(",") if m.strip()) if len(parts) > 1 and parts[1].strip() else set()
                gt_map[eid] = matches
                
    total_true_matches = sum(len(m) for m in gt_map.values())
    print(f"Ground truth has {total_true_matches:,d} true matches for this sample", flush=True)
    
    # 3. Index S2 and S3 for target_country
    t0 = time.time()
    name_idx = defaultdict(lambda: array("i"))
    addr_idx = defaultdict(lambda: array("i"))
    digit_idx = defaultdict(lambda: array("i"))
    
    # Map integer index to eid string
    target_id_list = []
    
    for target_file in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        with open(target_file, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    eid, name, addr, cntry = parts[0], parts[1], parts[2], parts[3].strip().upper()
                    if cntry == target_country:
                        t_int = len(target_id_list)
                        target_id_list.append(eid)
                        
                        for tok in get_name_tokens(name):
                            name_idx[tok].append(t_int)
                        for tok in get_addr_tokens(addr):
                            addr_idx[tok].append(t_int)
                        for dig in get_digits(addr):
                            digit_idx[dig].append(t_int)
                            
    print(f"Indexed {len(target_id_list):,d} target records in {time.time() - t0:.2f}s", flush=True)
    
    # Prune ultra frequent keys (> 2000 records)
    for idx in [name_idx, addr_idx, digit_idx]:
        keys_to_del = [k for k, v in idx.items() if len(v) > 2000]
        for k in keys_to_del:
            del idx[k]
            
    # 4. Query S1
    t1 = time.time()
    captured_tp = 0
    total_candidates = 0
    
    for s1_id, name, addr in s1_sample:
        cands_int = set()
        for tok in get_name_tokens(name):
            if tok in name_idx:
                cands_int.update(name_idx[tok])
        for dig in get_digits(addr):
            if dig in digit_idx:
                cands_int.update(digit_idx[dig])
        for tok in get_addr_tokens(addr):
            if tok in addr_idx:
                cands_int.update(addr_idx[tok])
                
        # Resolve to strings
        cand_strings = {target_id_list[idx] for idx in cands_int}
        true_m = gt_map.get(s1_id, set())
        captured = true_m.intersection(cand_strings)
        captured_tp += len(captured)
        total_candidates += len(cand_strings)
        
    query_time = time.time() - t1
    recall = (captured_tp / total_true_matches) if total_true_matches > 0 else 1.0
    avg_cands = total_candidates / len(s1_sample)
    
    print(f"Recall              : {recall * 100:.2f}% ({captured_tp:,d} / {total_true_matches:,d} true matches)", flush=True)
    print(f"Avg Candidates / S1 : {avg_cands:.1f}", flush=True)
    print(f"Throughput          : {len(s1_sample)/query_time:.1f} S1 entities/second", flush=True)


if __name__ == "__main__":
    run_country_benchmark("US", s1_limit=10000)
    run_country_benchmark("INDIA", s1_limit=10000)
