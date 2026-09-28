"""
High-Performance Pairwise Feature Extraction Engine for Amazon ML Challenge 2026.
Uses RapidFuzz (C++ accelerated) and token/digit overlap metrics for candidate scoring.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
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

    def __init__(self, use_expanded: bool = False):
        self.use_expanded = use_expanded
        self.feature_names = list(EXPANDED_FEATURE_COLUMNS if use_expanded else FEATURE_COLUMNS)

    def extract_pair_features(
        self,
        s1: Dict[str, Any],
        cand: Dict[str, Any],
        blocker_score: float = 0.0,
        blocker_rank: int = 1,
        name_idf_dict: Optional[Dict[str, float]] = None,
    ) -> List[float]:
        if self.use_expanded:
            return extract_expanded_pair_features(
                s1, cand, blocker_score=blocker_score, blocker_rank=blocker_rank, name_idf_dict=name_idf_dict
            )
        return extract_pair_features(s1, cand)


from rapidfuzz.distance import JaroWinkler


def char_ngram_cosine(s1: str, s2: str, n: int = 3) -> float:
    """Computes character n-gram cosine similarity."""
    clean1 = "".join([c for c in s1.lower() if c.isalnum()])
    clean2 = "".join([c for c in s2.lower() if c.isalnum()])
    if not clean1 and not clean2:
        return 1.0
    if not clean1 or not clean2 or len(clean1) < n or len(clean2) < n:
        return 1.0 if clean1 == clean2 else 0.0
    
    ng1 = [clean1[i:i+n] for i in range(len(clean1) - n + 1)]
    ng2 = [clean2[i:i+n] for i in range(len(clean2) - n + 1)]
    set1, set2 = set(ng1), set(ng2)
    inter = len(set1.intersection(set2))
    denom = (len(set1) * len(set2)) ** 0.5
    return inter / denom if denom > 0 else 0.0


EXPANDED_FEATURE_COLUMNS = list(FEATURE_COLUMNS) + [
    "name_jaro_winkler",
    "name_char3_cosine",
    "name_char4_cosine",
    "name_prefix3_match",
    "name_prefix5_match",
    "name_prefix8_match",
    "name_contain_s1_in_s2",
    "name_contain_s2_in_s1",
    "addr_contain_s1_in_s2",
    "addr_contain_s2_in_s1",
    "exact_postal_match",
    "leading_house_number_match",
    "high_name_high_addr",
    "high_name_low_addr",
    "low_name_high_addr",
    "low_name_low_addr",
    "candidate_score",
    "candidate_rank",
]


def extract_expanded_pair_features(
    s1: Dict[str, Any],
    cand: Dict[str, Any],
    blocker_rank: int = 1,
    blocker_score: float = 0.0,
    name_idf_dict: Optional[Dict[str, float]] = None,
) -> List[float]:
    """
    Extracts high-dimensional pairwise feature vector (39 features).
    """
    base_feats = extract_pair_features(s1, cand)

    n1_raw = s1.get("name_raw", "")
    n2_raw = cand.get("name_raw", "")
    n1_std = s1.get("name_std", "")
    n2_std = cand.get("name_std", "")
    n1_toks = s1.get("name_tokens", [])
    n2_toks = cand.get("name_tokens", [])

    a1_std = s1.get("addr_std", "")
    a2_std = cand.get("addr_std", "")
    a1_toks = s1.get("addr_tokens", [])
    a2_toks = cand.get("addr_tokens", [])
    d1 = s1.get("addr_digits", [])
    d2 = cand.get("addr_digits", [])

    # 1. Jaro-Winkler
    jw = JaroWinkler.similarity(n1_std, n2_std) if (n1_std and n2_std) else 0.0

    # 2. Character 3-gram and 4-gram cosine
    char3_cos = char_ngram_cosine(n1_std, n2_std, n=3)
    char4_cos = char_ngram_cosine(n1_std, n2_std, n=4)

    # 3. Prefix agreement (3, 5, 8 characters)
    p3 = 1.0 if (len(n1_std) >= 3 and len(n2_std) >= 3 and n1_std[:3] == n2_std[:3]) else 0.0
    p5 = 1.0 if (len(n1_std) >= 5 and len(n2_std) >= 5 and n1_std[:5] == n2_std[:5]) else 0.0
    p8 = 1.0 if (len(n1_std) >= 8 and len(n2_std) >= 8 and n1_std[:8] == n2_std[:8]) else 0.0

    # 4. Token containment in both directions
    inter_name = len(set(n1_toks).intersection(set(n2_toks)))
    name_contain_s1 = inter_name / len(n1_toks) if n1_toks else 0.0
    name_contain_s2 = inter_name / len(n2_toks) if n2_toks else 0.0

    # 5. Address token containment in both directions
    inter_addr = len(set(a1_toks).intersection(set(a2_toks)))
    addr_contain_s1 = inter_addr / len(a1_toks) if a1_toks else 0.0
    addr_contain_s2 = inter_addr / len(a2_toks) if a2_toks else 0.0

    # 6. Postal & House numbers
    pins1 = [d for d in d1 if len(d) in (5, 6)]
    pins2 = [d for d in d2 if len(d) in (5, 6)]
    if pins1 and pins2:
        exact_postal = 1.0 if bool(set(pins1).intersection(set(pins2))) else 0.0
    else:
        exact_postal = 0.5

    if d1 and d2:
        leading_house = 1.0 if d1[0] == d2[0] else 0.0
    else:
        leading_house = 0.5

    # 7. Name / Address Interaction Quadrants
    name_sim = base_feats[0]  # name_ratio
    addr_sim = base_feats[11] if len(base_feats) > 11 else 0.0  # addr_ratio
    high_name_high_addr = 1.0 if (name_sim >= 0.75 and addr_sim >= 0.70) else 0.0
    high_name_low_addr = 1.0 if (name_sim >= 0.75 and addr_sim < 0.40) else 0.0
    low_name_high_addr = 1.0 if (name_sim < 0.40 and addr_sim >= 0.70) else 0.0
    low_name_low_addr = 1.0 if (name_sim < 0.40 and addr_sim < 0.40) else 0.0

    # 8. Blocker metadata
    cand_score_norm = float(min(blocker_score / 30.0, 1.0))
    cand_rank_norm = float(1.0 / blocker_rank) if blocker_rank > 0 else 1.0

    return base_feats + [
        jw,
        char3_cos,
        char4_cos,
        p3,
        p5,
        p8,
        name_contain_s1,
        name_contain_s2,
        addr_contain_s1,
        addr_contain_s2,
        exact_postal,
        leading_house,
        high_name_high_addr,
        high_name_low_addr,
        low_name_high_addr,
        low_name_low_addr,
        cand_score_norm,
        cand_rank_norm,
    ]


