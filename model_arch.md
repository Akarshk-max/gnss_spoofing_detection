## 3. Model Architecture & Training Methodology

### 3.1 Model Choice: XGBoost (Tree Ensemble)
We selected **XGBoost** as our primary model for Round-1 submission.  

**Why XGBoost?**  
- Superior performance on tabular receiver observables (PRN, Doppler, Pseudorange, correlators, etc.).  
- Natively handles non-linear physics relationships and feature interactions.  
- Built-in feature importance for explainability (critical for Round-2 presentation and “research depth” criterion).  
- Extremely fast training (seconds to minutes on 891k rows), allowing rapid iteration before the 11:59 PM deadline.  
- Excellent support for imbalanced classification via `scale_pos_weight`.

This aligns perfectly with the competition’s permitted approaches (classical ML + hybrid signal-processing features) and the tabular nature of the dataset.

### 3.2 Training Pipeline & Key Configurations

```python
# Core settings used
xgb_params = {
    "tree_method": "hist",          # Fast histogram-based splitting
    "device": "cuda",               # Full GPU acceleration (80 GB VRAM)
    "eval_metric": "logloss",
    "early_stopping_rounds": 50
}
