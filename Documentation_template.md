# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** CODEX  
**Team Members:** HIMA KALYAN RAM KANDANA (LEADER), KAUSTHUB REDDY KARRE, MAHA LAKSHMI GAUIDE, KARTHIKEY SABBITHI  
**Submission Date:** 27-09-2026  

---

## 1. Executive Summary
We developed a scalable, high-precision, and recall-preserving Business Entity Resolution system designed to match 1.73M reference entities (`Source 1`) against 9.96M candidate records (`Source 2` and `Source 3`) across heterogeneous multi-country distributions (United States, India, and open-set France). Our architecture combines multi-channel inverted index blocking, high-speed C++ accelerated string & token feature extraction (`rapidfuzz`), and a LightGBM Gradient Boosted Decision Tree (GBDT) model calibrated with a dual-threshold decision engine targeting macro-averaged $F_{0.5}$. On our validation set of 15,000 holdout $S_1$ entities, our pipeline achieves a **Macro $F_{0.5}$ of 0.9098**, with **95.45% Precision**, **88.00% Recall**, and **88.85% Singleton Accuracy**.

---

## 2. Methodology

### 2.1 Problem Analysis
During extensive Exploratory Data Analysis (EDA) on the 2.2M training entities (7.64M true pairs), we uncovered several key domain challenges:
1. **Multi-Match & Singleton Topology:** An $S_1$ entity matches an average of 3.46 target records, but 5.58% of entities are true singletons (0 matches in $S_2/S_3$). Under macro-$F_{0.5}$, falsely predicting a match for a singleton yields an $F_{0.5}$ penalty of 0.0, making precision on singletons disproportionately impactful.
2. **Noise & Formatting Discrepancies:** Name variations encompass legal suffixes (`Pvt Ltd`, `LLC`, `Corp`), acronym abbreviations, spacing errors, and phonetic corruptions. Addresses exhibit heavy permutations, abbreviated street types (`RD`, `ST`, `BLVD`), omitted unit numbers, and disparate regional conventions between US, Indian PIN codes, and French department codes.
3. **Zero-Shot / Open-Set Country Generalization:** While training contains only US and India, the test partition introduces France (259k $S_1$ entities). Country handling must therefore be strictly partition-invariant, dynamic, and rule-based rather than memorized.

### 2.2 Solution Strategy
We structured our system as an end-to-end modular pipeline adhering to the competition constraints:

```
Test S1 Records (1.73M)
       ↓
Partitioning by Country (US / INDIA / FRANCE)
       ↓
Multi-Channel Inverted Index Blocking (Name Tokens, 6-Char Prefix, Address Digits, Street Tokens)
       ↓
Union & Candidate Scoring (Top-25 Candidates per S1) ───► output/candidate_pairs.tsv
       ↓
Pairwise Feature Extraction (15 String, Token, Digit & Interaction Features via RapidFuzz)
       ↓
LightGBM GBDT Pair Classifier (Probability Estimation)
       ↓
Dual-Threshold S1-Level Decision Engine (tau_singleton=0.80, tau_match=0.70) ───► output/matching_results.tsv
```

- **Approach Type:** Multi-Channel Inverted Blocking + GBDT Classifier + Macro-$F_{0.5}$ Calibrated Decision Engine.
- **Core Innovation:** Country-partitioned streaming batch inference with inverted index caching, coupled with an asymmetrical dual-threshold decision rule ($\tau_{singleton}, \tau_{match}$) specifically tailored to maximize the competition's macro-$F_{0.5}$ metric while strictly maintaining the subset invariant $\text{matching\_results} \subseteq \text{candidate\_pairs}$.

---

## 3. Candidate Generation (Blocking)

To reduce the $1.73\text{M} \times 9.96\text{M} \approx 17.2\text{ Trillion}$ pairwise comparison space into a sub-linear candidate set without losing true positive pairs, we engineered a multi-channel inverted index:

- **Blocking keys used:**
  1. **Cleaned Name Tokens:** Alphabetic tokens ($\ge 2$ characters), excluding stop words and corporate designators (`pvt`, `ltd`, `inc`, `corp`, `llc`, `services`, `enterprises`).
  2. **Compressed Name Prefix (6 chars):** Alphanumeric prefix capturing root business names across spelling variants and suffixes.
  3. **Address Numeric Tokens:** House numbers, building numbers, and postal codes ($\ge 2$ digits).
  4. **Street / Locality Tokens:** Informative address tokens ($\ge 4$ characters).
  5. **Country Partitioning:** Strict partition matching ensuring zero cross-country noise.
  6. **Frequency Pruning:** Tokens appearing in $>2,500$ records are pruned to eliminate high-entropy generic terms.

- **Candidate pairs generated:** Top-25 scored candidates per $S_1$ entity, yielding exactly 25 candidate target IDs per entity in `output/candidate_pairs.tsv`.
- **Recall Preservation:** On the training set ground truth, this multi-channel inverted index achieved **94.96% recall in US** and **94.12% recall in India**, ensuring minimal candidate attrition before classifier scoring.

---

## 4. Matching Model

### Features Used (15 Engineered Features):
1. `name_fuzz_ratio`: Full string Levenshtein similarity ratio between cleaned names.
2. `name_token_sort_ratio`: Word-order invariant token sort similarity ratio.
3. `name_token_set_ratio`: Set-based intersection similarity ratio (handles token insertions/deletions).
4. `name_partial_ratio`: Substring alignment similarity for abbreviated names.
5. `name_len_ratio`: Ratio of shortest to longest name length.
6. `name_prefix_match`: Binary indicator whether the first 4 characters of names match exactly.
7. `addr_fuzz_ratio`: Full string Levenshtein similarity ratio between cleaned addresses.
8. `addr_token_set_ratio`: Set-based token intersection similarity ratio for addresses.
9. `addr_partial_ratio`: Substring alignment similarity for addresses.
10. `digit_overlap_count`: Count of matching numeric sequences (house numbers, PIN codes).
11. `digit_jaccard`: Jaccard similarity index of address digit sets.
12. `digit_len_diff`: Absolute difference in count of numeric tokens.
13. `addr_len_ratio`: Ratio of shortest to longest address length.
14. `interaction_name_addr`: Product of `name_token_sort_ratio` and `addr_token_set_ratio`.
15. `score_max`: Maximum similarity score across all name and address features.

### Model Architecture & Training:
- **Model Type:** LightGBM Gradient Boosted Decision Tree (`LGBMClassifier`).
- **Hyperparameters:** `n_estimators=200`, `learning_rate=0.08`, `num_leaves=63`, `min_child_samples=30`, `subsample=0.85`, `colsample_bytree=0.85`.
- **Training Set Construction:** 400,000 balanced pairs sampled from the 2.2M training records (200,000 true positive matches and 200,000 hard negative candidate pairs generated by the blocking system).
- **Threshold Optimization:** Dual-threshold grid search on 15,000 validation entities:
  - $\tau_{\text{singleton}} = 0.80$: An $S_1$ entity is declared a singleton if $\max_{c} P(\text{match}|S_1, c) < 0.80$.
  - $\tau_{\text{match}} = 0.70$: For non-singleton entities, any candidate with $P(\text{match}|S_1, c) \ge 0.70$ is included in `matching_results.tsv`.

---

## 5. Results & Error Analysis

### Validation Performance (15,000 Holdout Entities):
- **Macro $F_{0.5}$ Score:** **0.909797** (0.9098)
- **Macro Precision:** **0.954457** (95.45%)
- **Macro Recall:** **0.879971** (88.00%)
- **Singleton Accuracy:** **88.85%** (804 / 905 singletons correctly identified)
- **Mean Predicted Matches per Non-Singleton:** 3.12 (vs 3.46 true matches)

### Error Analysis:
- **False Positives (Wrong Merges):**
  - Chain stores / Franchise branches sharing identical brand names in the same metropolitan area but having distinct suite/street numbers.
  - Sibling corporate entities with identical parent names differing only by fine-grained division names (`Trading Corp` vs `Securities Corp`).
- **False Negatives (Missed Matches):**
  - Extreme abbreviation shifts where legal names differ substantially from trade names (e.g., `SBI` vs `State Bank of India` with missing address tokens).
  - Severe typographic noise in Indian rural address structures where locality names are transliterated with distinct phonetic variants.

---

## 6. Conclusion
Our entity resolution pipeline delivers a robust, highly scalable, and accurate solution for large-scale multi-source business linkage. By pairing multi-channel inverted index blocking with rapid C++ feature extraction, GBDT scoring, and metric-calibrated dual thresholding, our architecture achieves **0.9098 Macro $F_{0.5}$** while executing within memory and compute bounds and ensuring full reproducibility across both known and unseen test partitions.

---

## Appendix

### A. Code Artefacts
All source code is structured under `code/business_entity_resolution/`:
```
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── eda_audit.py          # Data profiling & noise distribution analysis
│   ├── blocking.py           # Multi-channel inverted indexing & candidate generator
│   ├── features.py           # RapidFuzz-accelerated pairwise feature extraction
│   ├── metrics.py            # Official macro-F0.5 metric & evaluation suite
│   ├── train.py              # LightGBM training & threshold optimization
│   └── predict.py            # Streaming country-partitioned batch test inference
├── models/
│   └── entity_resolution_lgb.pkl   # Serialized LightGBM model artifact & thresholds
├── output/
│   ├── matching_results.tsv  # Final predicted matches per test S1 entity
│   └── candidate_pairs.tsv   # Full candidate set passed to the model
├── requirements.txt          # Minimal Python dependencies
└── README.md                 # Complete reproduction instructions
```

### B. Reproduction Commands
1. **Train Model & Tune Thresholds:**
   ```powershell
   py src/train.py --dataset_dir dataset --n_pos 200000 --n_neg 200000 --n_val 15000
   ```
2. **Generate Test Predictions:**
   ```powershell
   py src/predict.py --dataset_dir dataset --model_path models/entity_resolution_lgb.pkl --output_dir output --top_k 25
   ```
3. **Validate Submission Files:**
   ```powershell
   py utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
   ```
