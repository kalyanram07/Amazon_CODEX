"""
Multi-Strategy Candidate Generation & Blocking Engine for Amazon ML Challenge 2026.
Uses Country Partitioning, Inverted Indexes on Name Tokens, Prefixes, Address Digits & Tokens.
"""

from collections import defaultdict
from array import array
from typing import Dict, List, Set, Tuple, Any, Optional
import pandas as pd


LEGAL_WORDS = {
    "pvt", "ltd", "private", "limited", "inc", "incorporated", "corp", "corporation",
    "co", "company", "llc", "llp", "plc", "and", "the", "center", "centre", "group",
    "services", "enterprise", "enterprises", "solutions", "tech", "technology",
    "technologies", "associates", "international", "india", "usa"
}


class CountryBlocker:
    """
    Inverted Index blocker tailored for a single country partition.
    Indexes target records (S2 and S3) and rapidly queries candidates for S1 entities.
    """

    def __init__(self, max_token_frequency: int = 2500, default_top_k: int = 25):
        self.max_token_freq = max_token_frequency
        self.default_top_k = default_top_k
        self.name_idx = defaultdict(lambda: array("i"))
        self.prefix_idx = defaultdict(lambda: array("i"))
        self.addr_token_idx = defaultdict(lambda: array("i"))
        self.digit_idx = defaultdict(lambda: array("i"))
        self.target_records: List[Dict[str, Any]] = []
        self.target_id_to_int: Dict[str, int] = {}

    def fit_records(self, target_records: List[Dict[str, Any]]) -> "CountryBlocker":
        """Indexes a list of preprocessed target dictionaries."""
        self.target_records = target_records
        for idx, rec in enumerate(target_records):
            tid = rec["entity_id"]
            self.target_id_to_int[tid] = idx

            # 1. Name tokens
            for tok in rec.get("name_tokens", []):
                if len(tok) >= 2 and tok not in LEGAL_WORDS:
                    self.name_idx[tok].append(idx)

            # 2. Name prefix
            pref = rec.get("name_prefix", "")
            if pref:
                self.prefix_idx[pref].append(idx)

            # 3. Address digits
            for dig in rec.get("addr_digits", []):
                self.digit_idx[dig].append(idx)

            # 4. Address tokens
            for tok in rec.get("addr_tokens", []):
                if len(tok) >= 4:
                    self.addr_token_idx[tok].append(idx)

        # Prune high frequency stop keys
        for idx_dict in [self.name_idx, self.prefix_idx, self.addr_token_idx, self.digit_idx]:
            keys_to_del = [k for k, v in idx_dict.items() if len(v) > self.max_token_freq]
            for k in keys_to_del:
                del idx_dict[k]

        return self

    def block_entity(self, s1: Dict[str, Any], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves candidate target records for a single S1 record.
        Returns list of candidate dictionaries.
        """
        k = top_k if top_k is not None else self.default_top_k
        scores = defaultdict(float)

        # 1. Name token scoring
        for tok in s1.get("name_tokens", []):
            if tok in self.name_idx:
                for tid_int in self.name_idx[tok]:
                    scores[tid_int] += 3.0

        # 2. Name prefix scoring
        pref = s1.get("name_prefix", "")
        if pref and pref in self.prefix_idx:
            for tid_int in self.prefix_idx[pref]:
                scores[tid_int] += 4.0

        # 3. Address digit scoring
        for dig in s1.get("addr_digits", []):
            if dig in self.digit_idx:
                for tid_int in self.digit_idx[dig]:
                    scores[tid_int] += 2.0

        # 4. Address token scoring
        for tok in s1.get("addr_tokens", []):
            if tok in self.addr_token_idx:
                for tid_int in self.addr_token_idx[tok]:
                    scores[tid_int] += 1.0

        if not scores:
            return []

        # Sort and select top K
        if len(scores) > k:
            top_ints = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
        else:
            top_ints = list(scores.keys())

        return [self.target_records[i] for i in top_ints]

    def block_entity_ids(self, s1: Dict[str, Any], top_k: Optional[int] = None) -> List[str]:
        """Returns list of candidate entity_id strings for a single S1 record."""
        cands = self.block_entity(s1, top_k=top_k)
        return [c["entity_id"] for c in cands]


def benchmark_blocking_recall(
    candidate_map: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    total_target_records: int,
) -> Dict[str, float]:
    """
    Computes candidate recall and reduction ratio.
    """
    total_true_pairs = 0
    captured_true_pairs = 0
    total_candidates_generated = 0
    total_s1 = len(ground_truth)

    for s1_id, true_matches in ground_truth.items():
        total_true_pairs += len(true_matches)
        cands = candidate_map.get(s1_id, set())
        total_candidates_generated += len(cands)
        captured_true_pairs += len(true_matches.intersection(cands))

    candidate_recall = (captured_true_pairs / total_true_pairs) if total_true_pairs > 0 else 1.0
    avg_cands_per_s1 = total_candidates_generated / total_s1 if total_s1 > 0 else 0.0
    total_possible_comparisons = total_s1 * total_target_records
    reduction_ratio = 1.0 - (total_candidates_generated / total_possible_comparisons) if total_possible_comparisons > 0 else 1.0

    return {
        "candidate_recall": round(candidate_recall, 6),
        "avg_candidates_per_s1": round(avg_cands_per_s1, 2),
        "candidate_reduction_ratio": round(reduction_ratio, 6),
        "total_true_pairs": total_true_pairs,
        "captured_true_pairs": captured_true_pairs,
        "total_candidates": total_candidates_generated,
    }
