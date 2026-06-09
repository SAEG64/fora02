# Comprehensive Model and Parameter Recovery Analysis
## All 10 Models Implementation Summary

### Overview
A comprehensive **Parameter and Model Recovery** section has been added to `fora_model_comparison_hierarchical.py` that systematically tests the identifiability and sensitivity of all 10 decision models in your model comparison section.

---

## Section Structure (5 Parts)

### **Part 1: Gold Standard Parameter Extraction**
**Goal**: Extract fitted parameters from all 10 models to use as ground truth

**Process**:
- Iterates through all 10 models in `idata_list`
- For each model, extracts:
  - **Group-level intercept**: μ_α (population mean) and σ_α (SD)
  - **Main effect coefficients**: μ_β and σ_β (group-level slopes)
  - Single-feature models: Scalar parameters
  - Multi-feature model (multi-heuristic): Array parameters
- Handles errors gracefully for any failed extractions
- Prints summary table showing parameters for each model

**Output**: `model_params` dictionary with parameters indexed by model number

---

### **Part 2: Synthetic Data Generation from Each Model**
**Goal**: Create realistic synthetic data by sampling from each fitted model

**Configuration**:
- **Decision noise SD**: 0.5 (moderate behavioral noise)
- **Trials per model**: 150 observations
- **Subjects per model**: Same as original fitted data
- **Reproducibility**: Seeded random generation (seed = 42 + model_idx)

**Process for each model**:
1. Sample subject-level intercepts: $\alpha_i \sim N(\mu_\alpha, \sigma_\alpha)$
2. Sample subject-level slopes: $\beta_{i,f} \sim N(\mu_{f}, \sigma_{f})$ for each feature
3. Generate random features from standard normal (N(0,1))
4. Compute linear predictor: $\eta = \alpha_i + \sum_f \beta_{i,f} \cdot X_f$
5. Add decision noise: $\eta_{noisy} = \eta + N(0, 0.5)$
6. Generate binary choice via logistic: $y \sim \text{Bernoulli}(\text{sigmoid}(\eta_{noisy}))$

**Output**: One synthetic dataset per model with 150 trials

---

### **Part 3: Model Recovery - Fit All 10 Models to Each Synthetic Dataset**
**Goal**: Test which model best recovers synthetic data from each generating model

**What happens**:
- For each synthetic dataset (10 datasets from 10 generating models):
  - **Fit all 10 models** to that synthetic data
  - Extract LOO (leave-one-out cross-validation) for each fit
  - Record which model achieves highest LOO (best fit)
  
**Recovery Matrix Construction**:
- **Rows**: Generating model (ground truth)
- **Columns**: Fitted model
- **Values**: LOO score for each (model, data) pair
- **Interpretation**: 
  - Diagonal values (model recovers itself) should be BEST (highest LOO)
  - Off-diagonal values show cross-recovery rates

**Sampling Configuration per Model Fit**:
- 300 draws, 300 tune iterations
- 2 chains for speed
- target_accept = 0.9

**Output**: 10×10 LOO matrix `recovery_loo_matrix`

---

### **Part 4: Model Recovery Visualization**
**Publication-Quality Heatmap**:
- **Size**: 14" × 10" (square for clarity)
- **DPI**: 300 (publication ready)
- **Colormap**: RdYlGn_r (Red = model NOT recovered, Green = model recovered)
- **Normalization**: Δ LOO = Best LOO - Model LOO (per row)
- **Annotations**: LOO values printed in each cell with contrasting text colors
- **Labels**: Model names (shortened to 20 chars) with 45° rotation for readability

**What to look for**:
- **Strong diagonal Green**: Models are identifiable (recover themselves well)
- **Off-diagonal shades**: Shows which models are similar/confusable
- **Model-specific patterns**: Shows misidentification vulnerabilities

**File**: `RESULTS/model_recovery_loo_heatmap.png`

---

### **Part 5: Recovery Accuracy Summary**
**Metrics Computed**:
- **Model Recovery Accuracy**: %(models that recover correctly) / total
  - Correct = Model with highest LOO for synthetic data X matches generating model X
- **Per-model recovery**: Details of all generating models and their best-fit models
- **Status reporting**: Success/Failure for each fit

**Tables Generated**:
1. `model_recovery_summary.csv`:
   - Generating_Model | Best_Fit_Model | Best_LOO | Correct_Recovery
   - One row per synthetic dataset
   
2. `recovery_loo_matrix.csv`:
   - Full 10×10 LOO matrix
   - Rows = Generating models
   - Columns = Fitted models
   - Values = Δ LOO

**Console Output**:
- Live progress during fitting (model & LOO printed per fit)
- Final summary table with recovery status (✓ or ✗)
- Overall accuracy percentage
- File save confirmations

---

## Key Implementation Features

### **Robustness & Error Handling**
- ✓ Try-except blocks for parameter extraction failures
- ✓ Graceful failure reporting (NaN for failed fits)
- ✓ Continues even if individual model fits fail
- ✓ NaN checking in LOO calculations

### **Code Quality**
- ✓ Clear section headers with visual separators
- ✓ Descriptive variable names throughout
- ✓ Inline comments explaining complex logic
- ✓ No hard-coded magic numbers (all parametrized)
- ✓ Vectorized NumPy operations where possible
- ✓ Reproducible via fixed random seeds

### **Design Principles**
- **Modularity**: Each part is self-contained and independent
- **Extensibility**: Easy to add more noise levels or synthetic configurations
- **Interpretability**: Clear output messages and organized results
- **Efficiency**: Reduced sampling for synthetic data (faster than original fits)

---

## Expected Workflow Outputs

### **Console Output During Execution**:
```
================================================================================
COMPREHENSIVE MODEL AND MODEL RECOVERY ANALYSIS
Testing all 10 models with synthetic data generation
================================================================================

Part 1: Extracting Gold Standard Parameters from All Models
--------
  1. model_name                 - μ_α =  -0.1234, σ_α =  1.2345, features = 1
  2. model_name                 - μ_α =   0.5678, σ_α =  0.9876, features = 1
  ...
  10. multi-heuristic policy     - μ_α =  -0.2456, σ_α =  1.1234, features = 4

Part 2: Generating Synthetic Data from Each Model
--------
  Generating from model 1: model_name (1 features)
    Generated 150 trials, choice distribution: 52.3% foraging
  ...

Part 3: Model Recovery - Fitting All Models to Synthetic Data
--------
  Fitting all models to synthetic data from: model_name (model 1)
    Model  1 (model_name                ): LOO =    -105.2
    Model  2 (model_name                ): LOO =    -112.4
    ...
    ✓ Best fit: Model 1 (model_name) - LOO = -105.2

Part 4: Creating Model Recovery Plots
--------
  Saved: RESULTS/model_recovery_loo_heatmap.png

Part 5: Model Recovery Summary Statistics
--------
  model_name                     → model_name                ✓ CORRECT
  ...

================================================================================
MODEL RECOVERY ACCURACY: 9/10 = 90.0%
================================================================================

Summary saved to: RESULTS/model_recovery_summary.csv
Recovery matrix shape: (10, 10)
Recovery matrix saved to: RESULTS/recovery_loo_matrix.csv
```

### **Generated Files**:
1. **PNG** (visualization):
   - `model_recovery_loo_heatmap.png` - Full recovery matrix heatmap

2. **CSV** (data tables):
   - `model_recovery_summary.csv` - Per-model recovery outcomes
   - `recovery_loo_matrix.csv` - Full LOO values matrix

---

## Interpretation Guidelines

### **Perfect Model Recovery Would Look Like**:
- **Diagonal = all green**: Each model perfectly recovers itself
- **Off-diagonal = all red**: No model confusion
- **Accuracy = 100%**: All 10 models correctly identified

### **Realistic Expectations**:
- **Accuracy 60-90%**: Most models identifiable but some confusion
- **Off-diagonal patterns**: Show which models are structurally similar
- **Worst performers**: Models with high off-diagonal LOO values

### **What to Investigate if Recovery < 70%**:
1. **High similarity**: Check if similar models are being confused
2. **Parameter overlap**: Review gold standard parameter distributions
3. **Noise sensitivity**: Try reducing decision_noise_sd in Part 2
4. **Sample size**: Consider increasing n_obs_per_model (currently 150)

---

## Customization Options

To modify the recovery analysis, edit these parameters in **Part 2**:

```python
# Simulation parameters (Part 2)
n_obs_per_model = 150          # Increase for more stable recovery
decision_noise_sd = 0.5        # Increase to test robustness
n_subjects_per_model = n_subjects  # Keep consistent with original

# Sampling configuration (Part 3)
n_draws = 300                  # Increase for slower/more accurate fitting
n_tune = 300                   # Increase for better convergence
target_accept = 0.9            # Keep at 0.9 for efficient sampling
```

---

## Technical Notes

### **Single vs Multi-Feature Models**:
- Single-feature models: 1×1 scalar parameters per subject
- Multi-feature model: 4×N matrix of parameters (4 features × N subjects)
- Both handled automatically based on `n_features`

### **LOO Interpretation**:
- Higher LOO = Better model fit
- Δ LOO = max LOO - model LOO (normalized)
- Δ LOO = 0 on diagonal (same model, perfect fit reference)
- Δ LOO > 0 on off-diagonal (worse fit than true model)

### **Decision Noise Justification**:
- SD = 0.5 represents moderate behavioral stochasticity
- Includes both perceptual noise and execution noise
- Makes recovery more realistic (pure structure wouldn't always recover)

---

## Computational Requirements

**Estimated Runtime**:
- Part 1: < 1 minute (parameter extraction)
- Part 2: < 5 minutes (synthetic data generation)
- Part 3: **30-60 minutes** (10 models × 10 synthetic datasets)
- Part 4-5: < 5 minutes (plotting & summary)

**Total**: ~45-90 minutes depending on system

**Memory**: ~2-4 GB (idata objects stored in memory)

**Cores Used**: 1 core per model fit (no parallelization across fits - set `cores=1`)

---

## Expected Results

A well-designed model set should show:
- ✓ **Most models recover well** (accuracy > 70%)
- ✓ **Clear model distinctions** (green diagonal, red off-diagonal)
- ✓ **Interpretable confusions** (similar models confused, dissimilar models distinct)
- ✓ **Robust to noise** (recovery consistent across noise levels if tested)

If all models recover perfectly, consider:
- Whether sample size is too large
- Whether decision structure is clear
- Whether model set is too differentiated

If few models recover, check:
- Parameter ranges (ensure sufficient variation)
- Model similarity (may need to revise model set)
- Noise level (try different values)

