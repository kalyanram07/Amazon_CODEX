"""
High-Performance Pairwise Feature Extraction Engine for Amazon ML Challenge 2026.
Uses RapidFuzz (C++ accelerated) and token/digit overlap metrics for candidate scoring.
"""

from typing import Dict, List, Set, Tuple, Any
import numpy as np
from rapidfuzz import fuzz


def jaccard_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """Computes Jaccard similarity between two token lists."""
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    inter = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return inter / union if union > 0 else 0.0


def digit_overlap_ratio(digits1: List[str], digits2: List[str]) -> float:
    """Calculates overlap ratio of numeric sequences (house numbers, postal codes)."""
    set1, set2 = set(digits1), set(digits2)
    if not set1 and not set2:
        return 0.5  # Neutral signal when neither record has numbers
    if not set1 or not set2:
        return 0.0
    inter = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return inter / union if union > 0 else 0.0


def leading_digit_match(digits1: List[str], digits2: List[str]) -> float:
    """Checks if the primary street/house number matches exactly."""
    if digits1 and digits2:
        return 1.0 if digits1[0] == digits2[0] else 0.0
    return 0.5


FEATURE_COLUMNS = [
    "name_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_jaccard",
    "name_shared_tokens",
    "name_len_diff",
    "name_len_ratio",
    "name_prefix_match",
    "name_exact_raw",
    "name_exact_std",
    "addr_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_jaccard",
    "addr_shared_tokens",
    "addr_digit_overlap",
    "addr_lead_digit",
    "has_both_addr",
    "has_empty_addr",
    "source_is_s2",
    "source_is_s3",
    "name_addr_interaction",
]


def extract_pair_features(s1: Dict[str, Any], cand: Dict[str, Any]) -> List[float]:
    """
    Extracts numerical feature vector for a single (S1, Candidate) pair.
    Returns a Python list of floats matching FEATURE_COLUMNS order.
    """
    n1_raw = s1.get("name_raw", "")
    n2_raw = cand.get("name_raw", "")
    n1_std = s1.get("name_std", "")
    n2_std = cand.get("name_std", "")
    n1_toks = s1.get("name_tokens", [])
    n2_toks = cand.get("name_tokens", [])
    n1_pref = s1.get("name_prefix", "")
    n2_pref = cand.get("name_prefix", "")

    a1_raw = s1.get("addr_raw", "")
    a2_raw = cand.get("addr_raw", "")
    a1_std = s1.get("addr_std", "")
    a2_std = cand.get("addr_std", "")
    a1_toks = s1.get("addr_tokens", [])
    a2_toks = cand.get("addr_tokens", [])
    d1 = s1.get("addr_digits", [])
    d2 = cand.get("addr_digits", [])
    cand_id = cand.get("entity_id", "")

    # 1. Name Features
    name_ratio = fuzz.ratio(n1_std, n2_std) / 100.0 if (n1_std or n2_std) else 0.0
    name_token_sort_ratio = fuzz.token_sort_ratio(n1_std, n2_std) / 100.0 if (n1_std or n2_std) else 0.0
    name_token_set_ratio = fuzz.token_set_ratio(n1_std, n2_std) / 100.0 if (n1_std or n2_std) else 0.0
    name_partial_ratio = fuzz.partial_ratio(n1_std, n2_std) / 100.0 if (n1_std or n2_std) else 0.0
    name_jacc = jaccard_similarity(n1_toks, n2_toks)
    name_shared = float(len(set(n1_toks).intersection(set(n2_toks))))
    name_len_diff = float(abs(len(n1_std) - len(n2_std)))
    name_len_ratio = float(min(len(n1_std), len(n2_std)) / max(len(n1_std), len(n2_std), 1))
    name_prefix_match = 1.0 if (n1_pref and n2_pref and n1_pref == n2_pref) else 0.0
    name_exact_raw = 1.0 if (n1_raw and n1_raw == n2_raw) else 0.0
    name_exact_std = 1.0 if (n1_std and n1_std == n2_std) else 0.0

    # 2. Address Features
    both_addr = bool(a1_std and a2_std)
    has_both_addr = 1.0 if both_addr else 0.0
    has_empty_addr = 1.0 if (not a1_std or not a2_std) else 0.0

    if both_addr:
        addr_ratio = fuzz.ratio(a1_std, a2_std) / 100.0
        addr_token_sort_ratio = fuzz.token_sort_ratio(a1_std, a2_std) / 100.0
        addr_token_set_ratio = fuzz.token_set_ratio(a1_std, a2_std) / 100.0
        addr_jacc = jaccard_similarity(a1_toks, a2_toks)
        addr_shared = float(len(set(a1_toks).intersection(set(a2_toks))))
    else:
        addr_ratio = 0.0
        addr_token_sort_ratio = 0.0
        addr_token_set_ratio = 0.0
        addr_jacc = 0.0
        addr_shared = 0.0

    addr_digit_overlap = digit_overlap_ratio(d1, d2)
    addr_lead_digit = leading_digit_match(d1, d2)

    # 3. Source indicators
    source_is_s2 = 1.0 if cand_id.startswith("S2") else 0.0
    source_is_s3 = 1.0 if cand_id.startswith("S3") else 0.0

    # 4. Interaction feature
    name_addr_interaction = name_token_set_ratio * (addr_token_set_ratio if both_addr else 0.5)

    return [
        name_ratio,
        name_token_sort_ratio,
        name_token_set_ratio,
        name_partial_ratio,
        name_jacc,
        name_shared,
        name_len_diff,
        name_len_ratio,
        name_prefix_match,
        name_exact_raw,
        name_exact_std,
        addr_ratio,
        addr_token_sort_ratio,
        addr_token_set_ratio,
        addr_jacc,
        addr_shared,
        addr_digit_overlap,
        addr_lead_digit,
        has_both_addr,
        has_empty_addr,
        source_is_s2,
        source_is_s3,
        name_addr_interaction,
    ]


class PairFeatureExtractor:
    """Wrapper class for feature extraction with schema preservation."""

    def __init__(self):
        self.feature_names = list(FEATURE_COLUMNS)

    def extract_pair_features(self, s1: Dict[str, Any], cand: Dict[str, Any]) -> List[float]:
        return extract_pair_features(s1, cand)
