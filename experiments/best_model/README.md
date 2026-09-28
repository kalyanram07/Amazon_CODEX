# Best Experimental Model: Soft-IDF Blocker + LightGBM GBDT

## Validated Metrics (14,999 Holdout S1 Entities)
* **Macro F0.5:** 0.931041
* **Precision:** 0.967900
* **Recall:** 0.903305
* **Singleton Accuracy:** 93.26%
* **Optimal tau_singleton:** 0.90
* **Optimal tau_match:** 0.80

## Key Innovation & Rationale
1. **Continuous Soft-IDF Blocking:** Replaces destructive hard token frequency pruning with continuous entropy weighting $\text{IDF}(t) = \log((N+1)/(df(t)+1)) + 1$.
2. **Candidate Recall Boost:** Candidate recall increased from 87.93% to 91.22% on the validation set (and +16.26% on the full 6.18M target database) without increasing candidate set overhead ($K=25$).
3. **Macro F0.5 Improvement:** Macro F0.5 jumped from baseline 0.9093 to **0.931041** on the exact same 14,999 validation S1 entities.
