# Business Entity Resolution Pipeline — Team CODEX

**Competition:** Amazon ML Challenge 2026  
**Authors:** Team CODEX  
**Date:** September 2026  

---

## 1. Overview
This repository contains the complete, self-contained, reproducible pipeline for the Business Entity Resolution task.
The pipeline resolves reference entities from `Source 1` against noisy, unstructured candidate records in `Source 2` and `Source 3` across multiple countries (US, India, France).

---

## 2. Directory Structure
```
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── eda_audit.py            # Dataset audit, noise analysis & singleton profiling
│   ├── blocking.py             # Multi-channel inverted index blocking & candidate generator
│   ├── features.py             # RapidFuzz-accelerated string, token, digit feature extraction
│   ├── metrics.py              # Macro-averaged F0.5 metric with exact singleton logic
│   ├── train.py                # Balanced pair construction & LightGBM GBDT training
│   └── predict.py              # Streaming country-partitioned batch inference engine
├── models/
│   └── entity_resolution_lgb.pkl   # Serialized LightGBM model artifact & optimal thresholds
├── requirements.txt            # Pinned environment dependencies
└── README.md                   # This replication guide
```

---

## 3. Environment & Dependencies

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- 16 GB+ RAM recommended

### Installation
Install the required packages using `pip`:
```bash
pip install -r requirements.txt
```

### Core Libraries
- `lightgbm>=4.3.0`
- `rapidfuzz>=3.8.0`
- `pandas>=2.2.0`
- `numpy>=1.26.0`
- `scikit-learn>=1.4.0`

---

## 4. End-to-End Execution Guide

### Step 1: Data Preparation
Ensure the competition dataset is placed in `dataset/` with the following structure:
```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

### Step 2: Exploratory Data Analysis & Audit (Optional)
Run the dataset profiler to inspect entity counts, noise distributions, and country splits:
```bash
py src/eda_audit.py --dataset_dir dataset
```

### Step 3: Model Training & Threshold Calibration
Construct 400,000 balanced pairs (200k positive matches, 200k hard negative candidates) and train the LightGBM GBDT pair classifier:
```bash
py src/train.py --dataset_dir dataset --n_pos 200000 --n_neg 200000 --n_val 15000
```
This evaluates macro-$F_{0.5}$ across a 2D threshold grid and saves the serialized model to `models/entity_resolution_lgb.pkl`.

### Step 4: Full Test Inference
Generate `candidate_pairs.tsv` and `matching_results.tsv` for all 1,732,544 test $S_1$ entities using country-partitioned streaming batch inference:
```bash
py src/predict.py --dataset_dir dataset --model_path models/entity_resolution_lgb.pkl --output_dir output --top_k 25
```

### Step 5: Official Validator Verification
Validate that all submission requirements, singleton formats, and subset constraints ($\text{matching} \subseteq \text{candidate}$) strictly pass:
```bash
py utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

---

## 5. Performance Metrics
On a holdout validation set of 15,000 $S_1$ entities (51,900 true pairs):
- **Macro $F_{0.5}$:** `0.9098`
- **Macro Precision:** `0.9545`
- **Macro Recall:** `0.8800`
- **Singleton Accuracy:** `88.85%`
- **Blocking Recall:** `94.54%`
