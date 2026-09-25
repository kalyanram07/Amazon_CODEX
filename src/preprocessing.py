"""
Multi-View Text Preprocessor for Amazon ML Challenge 2026.
Generates non-destructive, parallel representations of business records:
- raw: original preserved casing & punctuation
- standardized: normalized legal suffixes (Pvt Ltd <-> Private Limited), road terms (Rd <-> Road, St <-> Street)
- tokens: filtered alphanumeric word sets for inverted indexing
- numbers: extracted numeric strings (PIN codes, house numbers, street digits)
- prefix: compressed alphanumeric prefix for domain and compact name matching
"""

import re
import string
from typing import Dict, List, Set, Any, Tuple
import pandas as pd

# Canonical legal suffix mappings
LEGAL_EXPANSIONS = {
    r"\bpvt\b": "private",
    r"\bltd\b": "limited",
    r"\binc\b": "incorporated",
    r"\bcorp\b": "corporation",
    r"\bco\b": "company",
    r"\bllc\b": "limited liability company",
    r"\bllp\b": "limited liability partnership",
    r"\bplc\b": "public limited company",
    r"\benterprise\b": "enterprises",
    r"\bsolution\b": "solutions",
    r"\btech\b": "technology",
    r"&": "and",
}

# Canonical address term mappings
ADDRESS_EXPANSIONS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bln\b": "lane",
    r"\bdr\b": "drive",
    r"\bflr\b": "floor",
    r"\bfl\b": "floor",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bno\b": "number",
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
    r"&": "and",
}

COMMON_STOP_WORDS = {
    "the", "a", "an", "and", "or", "of", "in", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "on", "off",
    "private", "limited", "pvt", "ltd", "inc", "corp", "corporation", "co",
    "company", "llc", "llp", "plc"
}


def normalize_whitespace(text: str) -> str:
    """Collapses consecutive spaces and strips outer whitespace."""
    return re.sub(r"\s+", " ", str(text)).strip()


def strip_punctuation(text: str) -> str:
    """Removes standard punctuation characters."""
    return text.translate(str.maketrans("", "", string.punctuation))


def clean_text_fast(text: str) -> str:
    """Fast lowercase punctuation strip and whitespace collapse."""
    if not text:
        return ""
    t = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(t.split())


def standardize_name(raw_name: str) -> str:
    """Standardizes legal suffixes and normalizes punctuation in business names."""
    if not raw_name or pd.isna(raw_name):
        return ""
    text = str(raw_name).lower()
    for pattern, replacement in LEGAL_EXPANSIONS.items():
        text = re.sub(pattern, replacement, text)
    text = strip_punctuation(text)
    return normalize_whitespace(text)


def standardize_address(raw_address: str) -> str:
    """Standardizes street terms and abbreviations in addresses."""
    if not raw_address or pd.isna(raw_address):
        return ""
    text = str(raw_address).lower()
    for pattern, replacement in ADDRESS_EXPANSIONS.items():
        text = re.sub(pattern, replacement, text)
    text = strip_punctuation(text)
    return normalize_whitespace(text)


def extract_tokens(standardized_text: str, filter_stops: bool = True) -> List[str]:
    """Extracts non-empty word tokens, optionally filtering stopwords and single chars."""
    if not standardized_text:
        return []
    tokens = [t for t in standardized_text.split() if t]
    if filter_stops:
        tokens = [t for t in tokens if t not in COMMON_STOP_WORDS and len(t) > 1]
    return tokens


def extract_digits(text: str) -> List[str]:
    """Extracts all contiguous numeric digit sequences (house numbers, postal codes)."""
    if not text or pd.isna(text):
        return []
    return re.findall(r"\b\d+\b", str(text))


def extract_name_prefix(name_std: str, min_len: int = 4, max_len: int = 8) -> str:
    """Extracts compressed alphanumeric prefix of business name for domain matching."""
    compact = "".join([c for c in name_std if c.isalnum()])
    if len(compact) >= min_len:
        return compact[:max_len]
    return ""


def preprocess_record(eid: str, name: str, addr: str, country: str) -> Dict[str, Any]:
    """Fast preprocessing of a single raw record tuple."""
    c_norm = normalize_whitespace(str(country).lower()) if country else ""
    n_raw = normalize_whitespace(name) if name else ""
    n_std = standardize_name(n_raw)
    n_toks = extract_tokens(n_std, filter_stops=True)
    n_pref = extract_name_prefix(n_std)

    a_raw = normalize_whitespace(addr) if addr else ""
    a_std = standardize_address(a_raw)
    a_toks = extract_tokens(a_std, filter_stops=True)
    a_digs = extract_digits(a_raw)

    return {
        "entity_id": eid,
        "country_norm": c_norm,
        "name_raw": n_raw,
        "name_std": n_std,
        "name_tokens": n_toks,
        "name_prefix": n_pref,
        "addr_raw": a_raw,
        "addr_std": a_std,
        "addr_tokens": a_toks,
        "addr_digits": a_digs,
    }


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies multi-view preprocessing to a dataframe.
    """
    df = df.copy()
    for col in ["entity_id", "business_name", "business_address", "country"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    df["country_norm"] = df["country"].apply(lambda c: normalize_whitespace(str(c).lower()))
    df["name_raw"] = df["business_name"].apply(normalize_whitespace)
    df["name_std"] = df["business_name"].apply(standardize_name)
    df["name_tokens"] = df["name_std"].apply(lambda s: extract_tokens(s, filter_stops=True))
    df["name_prefix"] = df["name_std"].apply(extract_name_prefix)

    df["addr_raw"] = df["business_address"].apply(normalize_whitespace)
    df["addr_std"] = df["business_address"].apply(standardize_address)
    df["addr_tokens"] = df["addr_std"].apply(lambda s: extract_tokens(s, filter_stops=True))
    df["addr_digits"] = df["business_address"].apply(extract_digits)

    return df
