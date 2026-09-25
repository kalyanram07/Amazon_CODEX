"""
Benchmark Top-K Candidate Pruning & Recall Trade-off.
"""
import time
import re
import string
from collections import defaultdict, Counter
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

print("Loading S1 and GT for US...")
s1_sample = []
s1_id_set = set()
with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            eid, name, addr, cntry = parts[0], parts[1], parts[2], parts[3].strip().upper()
            if cntry == "US":
                s1_sample.append((eid, name, addr))
                s1_id_set.add(eid)
                if len(s1_sample) >= 5000:
                    break

gt_map = {}
with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.strip().split("\t")
        eid = parts[0]
        if eid in s1_id_set:
            matches = set(m.strip() for m in parts[1].split(",") if m.strip()) if len(parts) > 1 and parts[1].strip() else set()
            gt_map[eid] = matches
            
total_tp = sum(len(m) for m in gt_map.values())

print("Indexing US targets...")
name_idx = defaultdict(lambda: array("i"))
addr_idx = defaultdict(lambda: array("i"))
digit_idx = defaultdict(lambda: array("i"))
target_id_list = []

for target_file in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
    with open(target_file, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                eid, name, addr, cntry = parts[0], parts[1], parts[2], parts[3].strip().upper()
                if cntry == "US":
                    t_int = len(target_id_list)
                    target_id_list.append(eid)
                    for tok in get_name_tokens(name):
                        name_idx[tok].append(t_int)
                    for tok in get_addr_tokens(addr):
                        addr_idx[tok].append(t_int)
                    for dig in get_digits(addr):
                        digit_idx[dig].append(t_int)

for idx in [name_idx, addr_idx, digit_idx]:
    keys_to_del = [k for k, v in idx.items() if len(v) > 2000]
    for k in keys_to_del:
        del idx[k]

print("Testing Top-K candidate recall thresholds...")
for K in [15, 25, 35, 50, 75]:
    captured = 0
    total_cands = 0
    t0 = time.time()
    for s1_id, name, addr in s1_sample:
        # Score candidates by token overlap weight
        scores = defaultdict(float)
        for tok in get_name_tokens(name):
            if tok in name_idx:
                for tid in name_idx[tok]:
                    scores[tid] += 3.0
        for dig in get_digits(addr):
            if dig in digit_idx:
                for tid in digit_idx[dig]:
                    scores[tid] += 1.5
        for tok in get_addr_tokens(addr):
            if tok in addr_idx:
                for tid in addr_idx[tok]:
                    scores[tid] += 1.0
                    
        # Sort top K
        if len(scores) > K:
            top_k_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:K]
        else:
            top_k_ids = list(scores.keys())
            
        cand_strings = {target_id_list[idx] for idx in top_k_ids}
        true_m = gt_map.get(s1_id, set())
        captured += len(true_m.intersection(cand_strings))
        total_cands += len(cand_strings)
        
    rec = captured / total_tp
    avg_c = total_cands / len(s1_sample)
    print(f"Top-{K:2d}: Recall = {rec*100:.2f}% ({captured:,d}/{total_tp:,d}) | Avg Cands/S1 = {avg_c:.1f} | Time: {time.time()-t0:.2f}s", flush=True)
