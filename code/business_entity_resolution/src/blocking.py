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


MultiStrategyBlocker = CountryBlocker


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


def extract_char3_ngrams(text: str) -> List[str]:
    """Extracts alphanumeric character 3-grams for noisy name retrieval."""
    clean = "".join([c for c in text.lower() if c.isalnum()])
    if len(clean) < 3:
        return [clean] if clean else []
    return [clean[i:i+3] for i in range(len(clean) - 2)]


class CountrySoftIDFBlocker:
    """
    Multi-Channel Soft-IDF Inverted Index Blocker with continuous entropy weighting
    and optional character 3-gram retrieval.
    Replaces binary token deletion with continuous IDF weighting log((N+1)/(df+1)) + 1.
    """

    def __init__(
        self,
        default_top_k: int = 25,
        enable_char3: bool = False,
        max_token_frequency: int = 3000,
    ):
        self.default_top_k = default_top_k
        self.enable_char3 = enable_char3
        self.max_token_freq = max_token_frequency

        self.name_idx = defaultdict(lambda: array("i"))
        self.prefix_idx = defaultdict(lambda: array("i"))
        self.digit_idx = defaultdict(lambda: array("i"))
        self.addr_token_idx = defaultdict(lambda: array("i"))
        self.char3_idx = defaultdict(lambda: array("i"))

        self.name_idf: Dict[str, float] = {}
        self.prefix_idf: Dict[str, float] = {}
        self.digit_idf: Dict[str, float] = {}
        self.addr_idf: Dict[str, float] = {}
        self.char3_idf: Dict[str, float] = {}

        self.target_records: List[Dict[str, Any]] = []
        self.target_id_to_int: Dict[str, int] = {}
        self.num_targets = 0

    def fit_records(self, target_records: List[Dict[str, Any]]) -> "CountrySoftIDFBlocker":
        """Indexes target dictionaries and computes exact corpus IDF distributions."""
        import math
        self.target_records = target_records
        self.num_targets = len(target_records)
        N = self.num_targets

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

            # 5. Optional Character 3-grams
            if self.enable_char3:
                name_raw = rec.get("name_raw", "")
                for c3 in extract_char3_ngrams(name_raw):
                    self.char3_idx[c3].append(idx)

        # Compute continuous IDF weights: log((N + 1) / (df + 1)) + 1
        self.name_idf = {k: math.log((N + 1.0) / (len(v) + 1.0)) + 1.0 for k, v in self.name_idx.items()}
        self.prefix_idf = {k: math.log((N + 1.0) / (len(v) + 1.0)) + 1.0 for k, v in self.prefix_idx.items()}
        self.digit_idf = {k: math.log((N + 1.0) / (len(v) + 1.0)) + 1.0 for k, v in self.digit_idx.items()}
        self.addr_idf = {k: math.log((N + 1.0) / (len(v) + 1.0)) + 1.0 for k, v in self.addr_token_idx.items()}
        if self.enable_char3:
            self.char3_idf = {k: math.log((N + 1.0) / (len(v) + 1.0)) + 1.0 for k, v in self.char3_idx.items()}

        # Prune high frequency stop keys
        for idx_dict in [self.name_idx, self.prefix_idx, self.digit_idx, self.addr_token_idx, self.char3_idx]:
            keys_to_del = [k for k, v in idx_dict.items() if len(v) > self.max_token_freq]
            for k in keys_to_del:
                del idx_dict[k]

        return self

    def block_entity_with_scores(
        self, s1: Dict[str, Any], top_k: Optional[int] = None
    ) -> List[Tuple[Dict[str, Any], float, int]]:
        """
        Retrieves top-K candidates with their blocking scores and ranks.
        Returns List of (candidate_dict, blocker_score, blocker_rank).
        """
        k = top_k if top_k is not None else self.default_top_k
        scores = defaultdict(float)

        # 1. Name tokens with continuous IDF
        for tok in s1.get("name_tokens", []):
            if tok in self.name_idx:
                w = 1.5 * min(self.name_idf.get(tok, 1.0), 10.0)
                for tid_int in self.name_idx[tok]:
                    scores[tid_int] += w

        # 2. Name prefix with continuous IDF
        pref = s1.get("name_prefix", "")
        if pref and pref in self.prefix_idx:
            w = 1.8 * min(self.prefix_idf.get(pref, 1.0), 10.0)
            for tid_int in self.prefix_idx[pref]:
                scores[tid_int] += w

        # 3. Address digits with continuous IDF
        for dig in s1.get("addr_digits", []):
            if dig in self.digit_idx:
                w = 1.2 * min(self.digit_idf.get(dig, 1.0), 8.0)
                for tid_int in self.digit_idx[dig]:
                    scores[tid_int] += w

        # 4. Address tokens with continuous IDF
        for tok in s1.get("addr_tokens", []):
            if tok in self.addr_token_idx:
                w = 0.8 * min(self.addr_idf.get(tok, 1.0), 6.0)
                for tid_int in self.addr_token_idx[tok]:
                    scores[tid_int] += w

        # 5. Optional Char 3-grams
        if self.enable_char3:
            name_raw = s1.get("name_raw", "")
            for c3 in extract_char3_ngrams(name_raw):
                if c3 in self.char3_idx:
                    w = 0.4 * min(self.char3_idf.get(c3, 1.0), 5.0)
                    for tid_int in self.char3_idx[c3]:
                        scores[tid_int] += w

        if not scores:
            return []

        # Sort and select top K
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [
            (self.target_records[idx], score, rank + 1)
            for rank, (idx, score) in enumerate(sorted_items)
        ]

    def block_entity(self, s1: Dict[str, Any], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Standard candidate dict list interface."""
        cands_with_scores = self.block_entity_with_scores(s1, top_k=top_k)
        return [c[0] for c in cands_with_scores]

    def block_entity_ids(self, s1: Dict[str, Any], top_k: Optional[int] = None) -> List[str]:
        """Returns candidate entity_id strings."""
        cands = self.block_entity(s1, top_k=top_k)
        return [c["entity_id"] for c in cands]


def extract_char4_ngrams(text: str) -> List[str]:
    """Extracts alphanumeric character 4-grams for high-precision noisy name retrieval."""
    clean = "".join([c for c in text.lower() if c.isalnum()])
    if len(clean) < 4:
        return [clean] if clean else []
    return [clean[i:i+4] for i in range(len(clean) - 3)]



class MultiChannelUnionBlocker:
    """
    7-Channel High-Recall Union Blocker for Business Entity Resolution.
    Combines:
    1. Exact Normalized Name Index
    2. Exact Normalized Address Index
    3. Soft-IDF Name Token Retrieval (Continuous Entropy Weighting)
    4. Character 3-Gram Index (Typo Tolerance)
    5. Character 4-Gram Index (Selective Substrings)
    6. Address Digit & Postal Code Index
    7. Rare Address Token Index
    """

    def __init__(
        self,
        default_top_k: int = 35,
        max_token_frequency: int = 5000,
        enable_char_ngrams: bool = True,
    ):
        self.default_top_k = default_top_k
        self.max_token_freq = max_token_frequency
        self.enable_char_ngrams = enable_char_ngrams

        # 7 Inverted Indexes
        self.exact_name_idx = defaultdict(lambda: array("i"))
        self.exact_addr_idx = defaultdict(lambda: array("i"))
        self.name_idx = defaultdict(lambda: array("i"))
        self.prefix_idx = defaultdict(lambda: array("i"))
        self.digit_idx = defaultdict(lambda: array("i"))
        self.addr_token_idx = defaultdict(lambda: array("i"))
        self.char3_idx = defaultdict(lambda: array("i"))
        self.char4_idx = defaultdict(lambda: array("i"))

        # IDF Dictionaries
        self.name_idf: Dict[str, float] = {}
        self.prefix_idf: Dict[str, float] = {}
        self.digit_idf: Dict[str, float] = {}
        self.addr_idf: Dict[str, float] = {}
        self.char3_idf: Dict[str, float] = {}
        self.char4_idf: Dict[str, float] = {}

        self.target_records: List[Dict[str, Any]] = []
        self.target_id_to_int: Dict[str, int] = {}
        self.num_targets = 0

    def fit_records(self, target_records: List[Dict[str, Any]]) -> "MultiChannelUnionBlocker":
        """Indexes target dictionaries across all 7 channels and computes corpus IDF distributions."""
        import math
        self.target_records = target_records
        self.num_targets = len(target_records)
        N = self.num_targets

        for idx, rec in enumerate(target_records):
            tid = rec["entity_id"]
            self.target_id_to_int[tid] = idx

            # Channel 1: Exact Normalized Name
            n_std = rec.get("name_std", "")
            if n_std:
                self.exact_name_idx[n_std].append(idx)

            # Channel 2: Exact Normalized Address
            a_std = rec.get("addr_std", "")
            if a_std:
                self.exact_addr_idx[a_std].append(idx)

            # Channel 3: Name Tokens
            for tok in rec.get("name_tokens", []):
                if len(tok) >= 2 and tok not in LEGAL_WORDS:
                    self.name_idx[tok].append(idx)

            # Prefix
            pref = rec.get("name_prefix", "")
            if pref:
                self.prefix_idx[pref].append(idx)

            # Channel 6: Address Digits & PIN/Postal codes
            for dig in rec.get("addr_digits", []):
                self.digit_idx[dig].append(idx)

            # Channel 7: Rare Address Tokens
            for tok in rec.get("addr_tokens", []):
                if len(tok) >= 4:
                    self.addr_token_idx[tok].append(idx)

            # Channels 4 & 5: Selective Character 4-grams for noisy name retrieval
            if self.enable_char_ngrams:
                name_raw = rec.get("name_raw", "")
                for c4 in extract_char4_ngrams(name_raw):
                    self.char4_idx[c4].append(idx)

        # Compute continuous IDF weights: log((N + 1) / (df + 1)) + 1
        self.name_idf = {k: math.log((N + 1.0) / (len(v) + 1.0)) + 1.0 for k, v in self.name_idx.items()}
        self.prefix_idf = {k: math.log((N + 1.0) / (len(v) + 1.0)) + 1.0 for k, v in self.prefix_idx.items()}
        self.digit_idf = {k: math.log((N + 1.0) / (len(v) + 1.0)) + 1.0 for k, v in self.digit_idx.items()}
        self.addr_idf = {k: math.log((N + 1.0) / (len(v) + 1.0)) + 1.0 for k, v in self.addr_token_idx.items()}
        if self.enable_char_ngrams:
            self.char4_idf = {k: math.log((N + 1.0) / (len(v) + 1.0)) + 1.0 for k, v in self.char4_idx.items()}

        # Prune high frequency stop keys
        for idx_dict in [self.name_idx, self.prefix_idx, self.digit_idx, self.addr_token_idx]:
            keys_to_del = [k for k, v in idx_dict.items() if len(v) > self.max_token_freq]
            for k in keys_to_del:
                del idx_dict[k]

        # Prune character 4-grams aggressively (keep only selective 4-grams with frequency <= 250)
        if self.enable_char_ngrams:
            keys_to_del = [k for k, v in self.char4_idx.items() if len(v) > 250]
            for k in keys_to_del:
                del self.char4_idx[k]

        return self

    def block_entity_with_scores(
        self, s1: Dict[str, Any], top_k: Optional[int] = None
    ) -> List[Tuple[Dict[str, Any], float, int]]:
        """
        Retrieves union candidates across all 7 channels with multi-signal score accumulation.
        Returns List of (candidate_dict, blocker_score, blocker_rank).
        """
        k = top_k if top_k is not None else self.default_top_k
        scores = defaultdict(float)

        # 1. Exact Name Matches (Massive boost)
        n_std = s1.get("name_std", "")
        if n_std and n_std in self.exact_name_idx:
            for tid_int in self.exact_name_idx[n_std]:
                scores[tid_int] += 30.0

        # 2. Exact Address Matches (High boost)
        a_std = s1.get("addr_std", "")
        if a_std and a_std in self.exact_addr_idx:
            for tid_int in self.exact_addr_idx[a_std]:
                scores[tid_int] += 20.0

        # 3. Name tokens with continuous IDF
        for tok in s1.get("name_tokens", []):
            if tok in self.name_idx:
                w = 1.8 * min(self.name_idf.get(tok, 1.0), 12.0)
                for tid_int in self.name_idx[tok]:
                    scores[tid_int] += w

        # 4. Name prefix
        pref = s1.get("name_prefix", "")
        if pref and pref in self.prefix_idx:
            w = 2.0 * min(self.prefix_idf.get(pref, 1.0), 10.0)
            for tid_int in self.prefix_idx[pref]:
                scores[tid_int] += w

        # 5. Address digits / PIN
        for dig in s1.get("addr_digits", []):
            if dig in self.digit_idx:
                w = 1.4 * min(self.digit_idf.get(dig, 1.0), 10.0)
                for tid_int in self.digit_idx[dig]:
                    scores[tid_int] += w

        # 6. Rare address tokens
        for tok in s1.get("addr_tokens", []):
            if tok in self.addr_token_idx:
                w = 0.9 * min(self.addr_idf.get(tok, 1.0), 8.0)
                for tid_int in self.addr_token_idx[tok]:
                    scores[tid_int] += w

        # 7. Character 4-grams (Selective fallback for difficult noisy names)
        if self.enable_char_ngrams and len(scores) < k:
            name_raw = s1.get("name_raw", "")
            for c4 in extract_char4_ngrams(name_raw):
                if c4 in self.char4_idx:
                    w = 0.8 * min(self.char4_idf.get(c4, 1.0), 8.0)
                    for tid_int in self.char4_idx[c4]:
                        scores[tid_int] += w

        if not scores:
            return []

        # Sort and select top K
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [
            (self.target_records[idx], score, rank + 1)
            for rank, (idx, score) in enumerate(sorted_items)
        ]

    def block_entity(self, s1: Dict[str, Any], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Standard candidate dict list interface."""
        cands_with_scores = self.block_entity_with_scores(s1, top_k=top_k)
        return [c[0] for c in cands_with_scores]

    def block_entity_ids(self, s1: Dict[str, Any], top_k: Optional[int] = None) -> List[str]:
        """Returns candidate entity_id strings."""
        cands = self.block_entity(s1, top_k=top_k)
        return [c["entity_id"] for c in cands]


