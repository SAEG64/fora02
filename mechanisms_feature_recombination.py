# %% Data Preprocessing
""" Prepare data for hierarchical mixture model with condition and block effects. """

import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import os

path = os.path.dirname(__file__) + "/"

# ==============================================================================
# 1. DATA PREPARATION
# ==============================================================================

df = pd.read_csv(path + 'DATA_clean/DATA_group_level/datall_with_condition_order.csv')

# ------------------------------------------------------------------------------
# SUBJECT INDEX
# ------------------------------------------------------------------------------

df["subj_idx"], subj_ids = pd.factorize(df["Subject_ID"])
n_subj = len(subj_ids)

# ------------------------------------------------------------------------------
# BLOCK CREATION (CORRECT: PRESERVE TRUE ORDER)
# ------------------------------------------------------------------------------

# Previous miniblock value per subject
prev_mb = df.groupby("Subject_ID")["miniblock.thisN"].shift()

# Detect FIRST 0 of each sequence
df["block_start"] = (
    (df["miniblock.thisN"] == 0) & (prev_mb != 0)
).astype(int)

# Count blocks per subject
df["block_idx"] = df.groupby("Subject_ID")["block_start"].cumsum() -1

# Clean up
df = df.drop(columns=["block_start"])

# IMPORTANT: Remove trials with missing outcome data (NaN in foraging T/F NaNs)
# These represent invalid trials that should be excluded from all analyses
df = df[df["foraging T/F NaNs"].notna()].copy()

# ------------------------------------------------------------------------------
# BLOCK VARIABLE (WITHIN-SUBJECT CENTERED)
# ------------------------------------------------------------------------------

df["block_centered"] = df.groupby("Subject_ID")["block_idx"].transform(
    lambda x: x - x.mean()
)
df["block_scaled"] = df.groupby("Subject_ID")["block_centered"].transform(
    lambda x: x / x.std()
)

block_z = df["block_scaled"].values

# ------------------------------------------------------------------------------
# FILTER DATA
# ------------------------------------------------------------------------------

df = df[df['multi-heuristic policy'] != 0]
df = df[df['multi-heuristic policy'] != 1]
print(f"\nData: {len(df.index)} Valid trials for modeling")

# %% Test Model
""" Hierarchical logistic regression for p_gain, r_threat, and their interaction. """
# ------------------------------------------------------------------------------
# VARIABLES
# ------------------------------------------------------------------------------
y = df["foraging T/F NaNs"].values.astype(int)  # Now safe - no NaNs remaining
# op = df["$\mathit{OP}$ values + cap"].values.astype(float)
op = df["OP_value_difference"].values.astype(float)
mh = df["multi-heuristic policy"].values.astype(float)

# Standardize predictors
op = (op - op.mean()) / op.std()
mh = (mh - mh.mean()) / mh.std()

# ------------------------------------------------------------------------------
# CONDITION CODING
# ------------------------------------------------------------------------------
cond = 1-pd.factorize(df["condition_rORp"])[0]
cond_signed = 2 * cond - 1   # -1 / +1

subj_idx = df["subj_idx"].values.astype(int)


# ==============================================================================
# 2. HIERARCHICAL MODEL: response ~ p_gain + r_threat + interaction
# ==============================================================================
"""
Tests the effect of probability gain, reward threat, and their interaction
on response, with random intercepts and slopes by participant.
"""

# Reload data for clean modeling
df_model = df.copy()

# Clean variable names and prepare data
df_model['p_gain'] = df_model['* $\\mathit{p}$ gain']
df_model['response'] = df_model['fora_response']

# Remove rows with missing values in key variables
df_clean = df_model[['participant', 'response', 'p_gain', 'r_threat', 'condition_rORp']].dropna()

# Standardize continuous predictors for better sampling
df_clean['p_gain_std'] = (df_clean['p_gain'] - df_clean['p_gain'].mean()) / df_clean['p_gain'].std()
df_clean['r_threat_std'] = (df_clean['r_threat'] - df_clean['r_threat'].mean()) / df_clean['r_threat'].std()

# Code condition as -1/+1 (for better interpretation)
df_clean['condition_code'] = cond_signed

# Create participant ID mapping
unique_participants = df_clean['participant'].unique()
participant_to_idx = {p: i for i, p in enumerate(unique_participants)}
df_clean['participant_idx'] = df_clean['participant'].map(participant_to_idx)

# Prepare data for modeling
y_pgain = df_clean['response'].values
X_p_gain = df_clean['p_gain_std'].values
X_r_threat = df_clean['r_threat_std'].values
X_condition = df_clean['condition_code'].values
X_p_r_interaction = X_p_gain * X_r_threat
X_p_condition = X_p_gain * X_condition
X_r_condition = X_r_threat * X_condition
X_three_way = X_p_gain * X_r_threat * X_condition  # Three-way interaction
participant_idx_pgain = df_clean['participant_idx'].values

n_obs = len(y_pgain)
n_participants_pgain = len(unique_participants)

print(f"\n{'='*70}")
print("HIERARCHICAL MODEL: response ~ p_gain + r_threat + condition +")
print("                    p_gain×r_threat + p_gain×condition + r_threat×condition +")
print("                    p_gain×r_threat×condition")
print(f"{'='*70}")
print(f"Observations: {n_obs}")
print(f"Participants: {n_participants_pgain}")
print(f"Response rate: {y_pgain.mean():.3f}")
print(f"P-Gain range: [{X_p_gain.min():.3f}, {X_p_gain.max():.3f}]")
print(f"R-Threat range: [{X_r_threat.min():.3f}, {X_r_threat.max():.3f}]")
print(f"Condition: {np.unique(X_condition)}")

# Build hierarchical model
coords_pgain = {"participant": unique_participants}

with pm.Model(coords=coords_pgain) as hierarchical_model:
    
    # Participant index
    participant_idx_pm = pm.Data("participant_idx", participant_idx_pgain)
    
    # =========================================================================
    # INTERCEPT: alpha = mu_alpha + individual_deviation
    # =========================================================================
    mu_alpha = pm.Deterministic("mu_alpha", pm.Normal("mu_alpha_raw", mu=0, sigma=1))
    sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)
    alpha_dev = pm.Normal("alpha_dev", mu=0, sigma=1, dims="participant")
    alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_dev, dims="participant")
    
    # =========================================================================
    # P_GAIN EFFECT: effect = mu_p_gain + individual_deviation
    # =========================================================================
    mu_p_gain = pm.Deterministic("mu_p_gain", pm.Normal("mu_p_gain_raw", mu=0, sigma=1))
    sigma_p_gain = pm.HalfNormal("sigma_p_gain", sigma=1)
    p_gain_dev = pm.Normal("p_gain_dev", mu=0, sigma=1, dims="participant")
    p_gain_effect = pm.Deterministic("p_gain_effect", mu_p_gain + sigma_p_gain * p_gain_dev, dims="participant")
    
    # =========================================================================
    # R_THREAT EFFECT: effect = mu_r_threat + individual_deviation
    # =========================================================================
    mu_r_threat = pm.Deterministic("mu_r_threat", pm.Normal("mu_r_threat_raw", mu=0, sigma=1))
    sigma_r_threat = pm.HalfNormal("sigma_r_threat", sigma=1)
    r_threat_dev = pm.Normal("r_threat_dev", mu=0, sigma=1, dims="participant")
    r_threat_effect = pm.Deterministic("r_threat_effect", mu_r_threat + sigma_r_threat * r_threat_dev, dims="participant")
    
    # =========================================================================
    # TWO-WAY INTERACTIONS (fixed, no individual variation)
    # =========================================================================
    beta_p_r_interaction = pm.Deterministic("beta_p_r_interaction", pm.Normal("beta_p_r_interaction_raw", mu=0, sigma=1))
    beta_p_condition = pm.Deterministic("beta_p_condition", pm.Normal("beta_p_condition_raw", mu=0, sigma=1))
    beta_r_condition = pm.Deterministic("beta_r_condition", pm.Normal("beta_r_condition_raw", mu=0, sigma=1))
    
    # =========================================================================
    # CONDITION EFFECT (fixed)
    # =========================================================================
    beta_condition = pm.Deterministic("beta_condition", pm.Normal("beta_condition_raw", mu=0, sigma=1))
    
    # =========================================================================
    # THREE-WAY INTERACTION: p_gain × r_threat × condition (fixed)
    # =========================================================================
    beta_three_way = pm.Deterministic("beta_three_way", pm.Normal("beta_three_way_raw", mu=0, sigma=1))
    
    # Linear predictor
    mu = (
        alpha[participant_idx_pm] +
        p_gain_effect[participant_idx_pm] * X_p_gain +
        r_threat_effect[participant_idx_pm] * X_r_threat +
        beta_condition * X_condition +
        beta_p_r_interaction * X_p_r_interaction +
        beta_p_condition * X_p_condition +
        beta_r_condition * X_r_condition +
        beta_three_way * X_three_way
    )
    
    # Likelihood
    p = pm.Deterministic("p", pm.math.sigmoid(mu))
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y_pgain)

print("\nModel structure:")
print(hierarchical_model)

# Fit model
print("\nFitting hierarchical model...")

with hierarchical_model:
    idata_pgain = pm.sample(
        draws=1000,
        tune=1000,
        target_accept=0.9,
        random_seed=42,
        progressbar=True,
        return_inferencedata=True
    )

print("\nSampling complete!")

# First, let's see what variables are actually in the posterior
print("\nDEBUG: Variables in posterior:")
print(sorted(idata_pgain.posterior.data_vars.keys()))

# Model diagnostics - intelligently extract whatever group-level parameters exist
all_vars = sorted(idata_pgain.posterior.data_vars.keys())
group_vars_wanted = ['mu_alpha', 'mu_p_gain', 'mu_r_threat', 
                     'beta_condition', 'beta_p_r_interaction', 'beta_p_condition', 'beta_r_condition', 'beta_three_way',
                     'sigma_alpha', 'sigma_p_gain', 'sigma_r_threat']
group_vars_available = [v for v in group_vars_wanted if v in all_vars]

print(f"\nGroup-level parameters requested: {group_vars_wanted}")
print(f"Group-level parameters available: {group_vars_available}")

if len(group_vars_available) == 0:
    print("\n⚠ ERROR: No group-level parameters found in posterior!")
    print("Available variables:", all_vars)
    summary_pgain = az.summary(idata_pgain)
else:
    summary_pgain = az.summary(idata_pgain, var_names=group_vars_available)

print("\nGroup-Level Parameters Summary:")
print(summary_pgain)
print(f"\nAvailable columns in summary: {summary_pgain.columns.tolist()}")

# Check Rhat values (should be < 1.01)
rhat_issues = summary_pgain[summary_pgain['r_hat'] > 1.01]
if len(rhat_issues) > 0:
    print("\n⚠ Warning: High Rhat values (convergence issues):")
    print(rhat_issues)
else:
    print("\n✓ All Rhat values <= 1.01 (good convergence)")

# Posterior predictive check
print("\nRunning posterior predictive checks...")
with hierarchical_model:
    idata_pgain.extend(pm.sample_posterior_predictive(idata_pgain))

# Results summary - Group-level parameters only
print("\n" + "="*70)
print("HIERARCHICAL MODEL RESULTS: p_gain + r_threat + condition +")
print("                            p_gain×r_threat + p_gain×condition + r_threat×condition +")
print("                            p_gain×r_threat×condition")
print("="*70)

# Group-level parameters to extract
group_level_params = ['mu_p_gain', 'mu_r_threat', 'beta_condition',
                      'beta_p_r_interaction', 'beta_p_condition', 'beta_r_condition', 'beta_three_way',
                      'mu_alpha', 'sigma_alpha', 'sigma_p_gain', 'sigma_r_threat']

print("\nGROUP-LEVEL PARAMETERS (Population-Level Estimates):")
print("-" * 70)
print(f"{'Parameter':<20} {'Mean':>10} {'95% HDI':<25} {'Rhat':>8}")
print("-" * 70)

# Find HDI columns - ArviZ uses different conventions
hdi_cols = [c for c in summary_pgain.columns if 'hdi' in c.lower()]
hdi_lows = sorted([c for c in hdi_cols if '%' in c and float(c.split('_')[1].replace('%', '')) < 50])
hdi_highs = sorted([c for c in hdi_cols if '%' in c and float(c.split('_')[1].replace('%', '')) > 50])

if hdi_lows and hdi_highs:
    hdi_low_col = hdi_lows[0]
    hdi_high_col = hdi_highs[-1]
    
    for var_name in group_level_params:
        if var_name in summary_pgain.index:
            mean = summary_pgain.loc[var_name, 'mean']
            hdi_low = summary_pgain.loc[var_name, hdi_low_col]
            hdi_high = summary_pgain.loc[var_name, hdi_high_col]
            rhat = summary_pgain.loc[var_name, 'r_hat']
            print(f"{var_name:<20} {mean:>10.4f} [{hdi_low:7.4f}, {hdi_high:7.4f}] {rhat:>8.4f}")
        else:
            print(f"{var_name:<20} {'⚠ Not found in summary':<10}")
else:
    # Fallback if HDI columns not found
    print("⚠ HDI columns not found in summary. Available columns:", summary_pgain.columns.tolist())
    for var_name in group_level_params:
        if var_name in summary_pgain.index:
            print(summary_pgain.loc[var_name])

# Check individual-level parameters (skip raw parameters and aggregate summaries)
print("\n" + "-" * 70)
all_summary = az.summary(idata_pgain)

# Individual intercepts
alpha_vars = [v for v in all_summary.index if v.startswith('alpha[')]
if len(alpha_vars) > 0:
    max_rhat = all_summary.loc[alpha_vars, 'r_hat'].max()
    if max_rhat <= 1.01:
        print(f"✓ Individual intercepts (alpha, n={len(alpha_vars)}): max Rhat = {max_rhat:.4f} [convergence OK]")
    else:
        bad_alpha = all_summary.loc[alpha_vars, 'r_hat'][all_summary.loc[alpha_vars, 'r_hat'] > 1.01]
        print(f"⚠ Individual intercepts: {len(bad_alpha)} with Rhat > 1.01")

# Individual p_gain effects
p_gain_effect_vars = [v for v in all_summary.index if v.startswith('p_gain_effect[')]
if len(p_gain_effect_vars) > 0:
    max_rhat = all_summary.loc[p_gain_effect_vars, 'r_hat'].max()
    if max_rhat < 1.01:
        print(f"✓ Individual p_gain effects (n={len(p_gain_effect_vars)}): max Rhat = {max_rhat:.4f} [convergence OK]")
    else:
        bad_slopes = all_summary.loc[p_gain_effect_vars, 'r_hat'][all_summary.loc[p_gain_effect_vars, 'r_hat'] > 1.01]
        print(f"⚠ Individual p_gain effects: {len(bad_slopes)} with Rhat > 1.01")

# Individual r_threat effects
r_threat_effect_vars = [v for v in all_summary.index if v.startswith('r_threat_effect[')]
if len(r_threat_effect_vars) > 0:
    max_rhat = all_summary.loc[r_threat_effect_vars, 'r_hat'].max()
    if max_rhat < 1.01:
        print(f"✓ Individual r_threat effects (n={len(r_threat_effect_vars)}): max Rhat = {max_rhat:.4f} [convergence OK]")
    else:
        bad_slopes = all_summary.loc[r_threat_effect_vars, 'r_hat'][all_summary.loc[r_threat_effect_vars, 'r_hat'] > 1.01]
        print(f"⚠ Individual r_threat effects: {len(bad_slopes)} with Rhat > 1.01")

# Main and interaction effects summary
print("\n" + "="*70)
print("MAIN AND INTERACTION EFFECTS SUMMARY")
print("="*70)

# Use the HDI columns we already identified
if hdi_lows and hdi_highs:
    hdi_low_col = hdi_lows[0]
    hdi_high_col = hdi_highs[-1]
    
    mu_p_gain_mean = summary_pgain.loc['mu_p_gain', 'mean']
    mu_p_gain_hdi_low = summary_pgain.loc['mu_p_gain', hdi_low_col]
    mu_p_gain_hdi_high = summary_pgain.loc['mu_p_gain', hdi_high_col]
    p_gain_includes_zero = mu_p_gain_hdi_low <= 0 <= mu_p_gain_hdi_high

    mu_r_threat_mean = summary_pgain.loc['mu_r_threat', 'mean']
    mu_r_threat_hdi_low = summary_pgain.loc['mu_r_threat', hdi_low_col]
    mu_r_threat_hdi_high = summary_pgain.loc['mu_r_threat', hdi_high_col]
    r_threat_includes_zero = mu_r_threat_hdi_low <= 0 <= mu_r_threat_hdi_high

    print(f"\n1. MAIN EFFECT: Probability Gain (p_gain)")
    print(f"   Posterior Mean: {mu_p_gain_mean:.4f}")
    print(f"   HDI: [{mu_p_gain_hdi_low:.4f}, {mu_p_gain_hdi_high:.4f}]")
    if p_gain_includes_zero:
        print(f"   ⚠ HDI includes zero: Effect is not clearly identified")
    else:
        direction = "positive" if mu_p_gain_mean > 0 else "negative"
        print(f"   ✓ Significant {direction} effect of p_gain on response")

    print(f"\n2. MAIN EFFECT: Relative Threat (r_threat)")
    print(f"   Posterior Mean: {mu_r_threat_mean:.4f}")
    print(f"   HDI: [{mu_r_threat_hdi_low:.4f}, {mu_r_threat_hdi_high:.4f}]")
    if r_threat_includes_zero:
        print(f"   ⚠ HDI includes zero: Effect is not clearly identified")
    else:
        direction = "positive" if mu_r_threat_mean > 0 else "negative"
        print(f"   ✓ Significant {direction} effect of r_threat on response")

    print(f"\n3. MAIN EFFECT: Condition")
    if 'beta_condition' in summary_pgain.index:
        beta_condition_mean = summary_pgain.loc['beta_condition', 'mean']
        beta_condition_hdi_low = summary_pgain.loc['beta_condition', hdi_low_col]
        beta_condition_hdi_high = summary_pgain.loc['beta_condition', hdi_high_col]
        condition_includes_zero = beta_condition_hdi_low <= 0 <= beta_condition_hdi_high
        print(f"   Posterior Mean: {beta_condition_mean:.4f}")
        print(f"   HDI: [{beta_condition_hdi_low:.4f}, {beta_condition_hdi_high:.4f}]")
        if condition_includes_zero:
            print(f"   ⚠ HDI includes zero: Effect is not clearly identified")
        else:
            direction = "positive" if beta_condition_mean > 0 else "negative"
            print(f"   ✓ Significant {direction} effect of condition on response")

    print(f"\n4. TWO-WAY INTERACTION: p_gain × r_threat")
    if 'beta_p_r_interaction' in summary_pgain.index:
        beta_p_r_mean = summary_pgain.loc['beta_p_r_interaction', 'mean']
        beta_p_r_hdi_low = summary_pgain.loc['beta_p_r_interaction', hdi_low_col]
        beta_p_r_hdi_high = summary_pgain.loc['beta_p_r_interaction', hdi_high_col]
        p_r_includes_zero = beta_p_r_hdi_low <= 0 <= beta_p_r_hdi_high
        print(f"   Posterior Mean: {beta_p_r_mean:.4f}")
        print(f"   HDI: [{beta_p_r_hdi_low:.4f}, {beta_p_r_hdi_high:.4f}]")
        if p_r_includes_zero:
            print(f"   ⚠ HDI includes zero: Interaction is not clearly identified")
        else:
            direction = "positive" if beta_p_r_mean > 0 else "negative"
            print(f"   ✓ Significant {direction} interaction between p_gain and r_threat")

    print(f"\n5. TWO-WAY INTERACTION: p_gain × condition")
    if 'beta_p_condition' in summary_pgain.index:
        beta_p_cond_mean = summary_pgain.loc['beta_p_condition', 'mean']
        beta_p_cond_hdi_low = summary_pgain.loc['beta_p_condition', hdi_low_col]
        beta_p_cond_hdi_high = summary_pgain.loc['beta_p_condition', hdi_high_col]
        p_cond_includes_zero = beta_p_cond_hdi_low <= 0 <= beta_p_cond_hdi_high
        print(f"   Posterior Mean: {beta_p_cond_mean:.4f}")
        print(f"   HDI: [{beta_p_cond_hdi_low:.4f}, {beta_p_cond_hdi_high:.4f}]")
        if p_cond_includes_zero:
            print(f"   ⚠ HDI includes zero: Interaction is not clearly identified")
        else:
            direction = "positive" if beta_p_cond_mean > 0 else "negative"
            print(f"   ✓ Significant {direction} interaction between p_gain and condition")

    print(f"\n6. TWO-WAY INTERACTION: r_threat × condition")
    if 'beta_r_condition' in summary_pgain.index:
        beta_r_cond_mean = summary_pgain.loc['beta_r_condition', 'mean']
        beta_r_cond_hdi_low = summary_pgain.loc['beta_r_condition', hdi_low_col]
        beta_r_cond_hdi_high = summary_pgain.loc['beta_r_condition', hdi_high_col]
        r_cond_includes_zero = beta_r_cond_hdi_low <= 0 <= beta_r_cond_hdi_high
        print(f"   Posterior Mean: {beta_r_cond_mean:.4f}")
        print(f"   HDI: [{beta_r_cond_hdi_low:.4f}, {beta_r_cond_hdi_high:.4f}]")
        if r_cond_includes_zero:
            print(f"   ⚠ HDI includes zero: Interaction is not clearly identified")
        else:
            direction = "positive" if beta_r_cond_mean > 0 else "negative"
            print(f"   ✓ Significant {direction} interaction between r_threat and condition")

    print(f"\n7. THREE-WAY INTERACTION: p_gain × r_threat × condition")
    if 'beta_three_way' in summary_pgain.index:
        beta_three_way_mean = summary_pgain.loc['beta_three_way', 'mean']
        beta_three_way_hdi_low = summary_pgain.loc['beta_three_way', hdi_low_col]
        beta_three_way_hdi_high = summary_pgain.loc['beta_three_way', hdi_high_col]
        three_way_includes_zero = beta_three_way_hdi_low <= 0 <= beta_three_way_hdi_high
        print(f"   Posterior Mean: {beta_three_way_mean:.4f}")
        print(f"   HDI: [{beta_three_way_hdi_low:.4f}, {beta_three_way_hdi_high:.4f}]")
        if three_way_includes_zero:
            print(f"   ⚠ HDI includes zero: Interaction is not clearly identified")
        else:
            direction = "positive" if beta_three_way_mean > 0 else "negative"
            print(f"   ✓ Significant {direction} three-way interaction")

    print(f"\nVariability across participants:")
    if 'sigma_alpha' in summary_pgain.index:
        print(f"   Intercept SD: {summary_pgain.loc['sigma_alpha', 'mean']:.4f}")
    if 'sigma_p_gain' in summary_pgain.index:
        print(f"   p_gain slope SD: {summary_pgain.loc['sigma_p_gain', 'mean']:.4f}")
    if 'sigma_r_threat' in summary_pgain.index:
        print(f"   r_threat slope SD: {summary_pgain.loc['sigma_r_threat', 'mean']:.4f}")
else:
    print("⚠ Could not extract HDI information from summary")

# %% Visualizations and Diagnostics
""" Create comprehensive set of visualizations for model diagnostics and results. """
# 

print("\n" + "="*70)
print("COMPREHENSIVE DIAGNOSTICS & VISUALIZATIONS")
print("="*70)

# Smart variable selection - only use variables that actually exist
available_vars = sorted(idata_pgain.posterior.data_vars.keys())

# Define variables to plot (population-level)
pop_level_vars = ["mu_alpha", "mu_p_gain", "mu_r_threat", "beta_p_r_interaction", 
                   "sigma_alpha", "sigma_p_gain", "sigma_r_threat"]
plot_var_names = [v for v in pop_level_vars if v in available_vars]
# Forest plot: all fixed effects + intercept variance
forest_var_names = [v for v in ["mu_alpha", "mu_p_gain", "mu_r_threat", "beta_condition",
                                  "beta_p_r_interaction", "beta_p_condition", "beta_r_condition", "beta_three_way",
                                  "sigma_alpha"] if v in available_vars]
rank_var_names = [v for v in ["mu_p_gain", "mu_r_threat", "beta_p_r_interaction"] if v in available_vars]
joint_var_names = [v for v in ["mu_p_gain", "mu_r_threat", "beta_p_r_interaction"] if v in available_vars]

print(f"\nVariables available for plotting: {plot_var_names}")

# 1. Trace plots (convergence visualization)
print("\n1. Generating trace plots...")
try:
    fig_trace = az.plot_trace(idata_pgain, var_names=plot_var_names, figsize=(14, 10))
    plt.suptitle("Trace Plots: Posterior Samples Over Iterations", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(path + "RESULTS/trace_plot_pgain_rthreat.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("   ✓ Saved: trace_plot_pgain_rthreat.png")
except Exception as e:
    print(f"   ⚠ Could not generate trace plot: {e}")

# 2. Forest plot (parameter estimates with credible intervals) - Publication quality
print("\n2. Generating forest plot...")
try:
    # Create mapping for cleaner parameter labels
    param_labels = {
        'mu_alpha': 'Intercept',
        'mu_p_gain': 'Probability Gain',
        'mu_r_threat': 'Relative Threat',
        'beta_condition': 'Condition',
        'beta_p_r_interaction': 'P.Gain × R.Threat',
        'beta_p_condition': 'P.Gain × Condition',
        'beta_r_condition': 'R.Threat × Condition',
        'beta_three_way': 'P.Gain × R.Threat × Condition',
        'sigma_alpha': 'Intercept SD'
    }
    
    # Generate forest plot with enhanced styling
    fig_forest = az.plot_forest(
        idata_pgain, 
        var_names=forest_var_names, 
        hdi_prob=0.95, 
        figsize=(12, 8),
        combined=True,
        linewidth=2.5
    )
    
    # Enhance plot aesthetics for publication
    ax = plt.gca()
    ax.set_xlabel('Effect Size (log-odds scale)', fontsize=12, fontweight='bold')
    
    # Add prominent zero reference line
    ax.axvline(x=0, color='darkred', linestyle='-', linewidth=2.5, alpha=0.8, label='Null (0)', zorder=5)
    
    # Add grid for easier reading
    ax.grid(True, alpha=0.4, linestyle='--', axis='x', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)  # Put grid behind the plot elements
    
    # Add legend for the zero line
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True)
    
    # Update title
    plt.suptitle('Forest Plot: Population-Level Parameters with 95% HDI', 
                 fontsize=13, fontweight='bold', y=0.98)
    
    # Improve layout
    plt.tight_layout()
    plt.savefig(path + "RESULTS/forest_plot_pgain_rthreat.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("   ✓ Saved: forest_plot_pgain_rthreat.png")
except Exception as e:
    print(f"   ⚠ Could not generate forest plot: {e}")

# 3. Posterior predictive check
print("\n3. Generating posterior predictive check...")
try:
    fig_ppc = az.plot_ppc(idata_pgain, figsize=(12, 5), num_pp_samples=100)
    plt.suptitle("Posterior Predictive Check", fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig(path + "RESULTS/ppc_pgain_rthreat.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("   ✓ Saved: ppc_pgain_rthreat.png")
except Exception as e:
    print(f"   ⚠ Could not generate PPC: {e}")

# 4. Posterior distributions
print("\n4. Generating posterior distributions...")
try:
    fig_post = az.plot_posterior(idata_pgain, var_names=plot_var_names, figsize=(14, 8), hdi_prob=0.95)
    plt.suptitle("Posterior Distributions (95% HDI)", fontsize=12, y=0.995)
    plt.tight_layout()
    plt.savefig(path + "RESULTS/posterior_dist_pgain_rthreat.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("   ✓ Saved: posterior_dist_pgain_rthreat.png")
except Exception as e:
    print(f"   ⚠ Could not generate posterior plot: {e}")

# 5. Rank plot (convergence diagnostic)
print("\n5. Generating rank plot...")
try:
    fig_rank = az.plot_rank(idata_pgain, var_names=rank_var_names, figsize=(12, 6))
    plt.suptitle("Rank Plot (Convergence Diagnostic)", fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig(path + "RESULTS/rank_plot_pgain_rthreat.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("   ✓ Saved: rank_plot_pgain_rthreat.png")
except Exception as e:
    print(f"   ⚠ Could not generate rank plot: {e}")

# 6. Energy plot (HMC diagnostics)
print("\n6. Generating energy plot...")
try:
    fig_energy = az.plot_energy(idata_pgain, figsize=(10, 6))
    plt.suptitle("HMC Energy Plot (Sampling Efficiency)", fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig(path + "RESULTS/energy_plot_pgain_rthreat.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("   ✓ Saved: energy_plot_pgain_rthreat.png")
except Exception as e:
    print(f"   ⚠ Could not generate energy plot: {e}")

# 7. Joint plot of key parameters
print("\n7. Generating joint parameter plot...")
try:
    fig_joint = az.plot_pair(idata_pgain, var_names=joint_var_names, kind='kde', figsize=(10, 10))
    plt.suptitle("Joint Posterior Distributions", fontsize=12, y=0.995)
    plt.tight_layout()
    plt.savefig(path + "RESULTS/joint_plot_pgain_rthreat.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("   ✓ Saved: joint_plot_pgain_rthreat.png")
except Exception as e:
    print(f"   ⚠ Could not generate joint plot: {e}")

# 8. Individual-level effects (if available)
print("\n8. Generating individual-level effects plots...")
try:
    indiv_effect_vars = [v for v in available_vars if any(x in v for x in ['alpha[', 'p_gain_effect[', 'r_threat_effect['])]
    if len(indiv_effect_vars) > 0:
        fig_indiv = az.plot_posterior(idata_pgain, var_names=indiv_effect_vars[:20], figsize=(14, 8), hdi_prob=0.95)  # Show first 20
        plt.suptitle("Individual Effects (First 20 Participants)", fontsize=12, y=0.995)
        plt.tight_layout()
        plt.savefig(path + "RESULTS/individual_effects_pgain_rthreat.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"   ✓ Saved: individual_effects_pgain_rthreat.png ({len(indiv_effect_vars)} total individual effects)")
except Exception as e:
    print(f"   ⚠ Could not generate individual effects plot: {e}")

print("\n✓ All visualizations completed!")
print("\nOutput files saved in: RESULTS/")

# Generate comprehensive summary table
print("\n" + "="*70)
print("POPULATION-LEVEL PARAMETERS: COMPREHENSIVE SUMMARY")
print("="*70)
# Dynamically select available columns
cols_to_keep = ['mean', 'std', 'r_hat', 'ess_bulk']
# Add HDI columns if they exist
hdi_cols_in_summary = [c for c in summary_pgain.columns if 'hdi' in c.lower()]
cols_to_keep.extend(hdi_cols_in_summary)
# Filter to only columns that actually exist
available_cols = [c for c in cols_to_keep if c in summary_pgain.columns]
summary_table = summary_pgain[available_cols].round(4)
print(summary_table.to_string())

print("\n" + "="*70)
print("INDIVIDUAL-LEVEL EFFECTS: SUMMARY STATISTICS")
print("="*70)
indiv_summary = all_summary.loc[[v for v in all_summary.index if any(x in v for x in ['alpha[', 'p_gain_effect[', 'r_threat_effect['])]]

if len(indiv_summary) > 0:
    # Compute counts outside f-string to avoid backslash issue
    alpha_count = len(indiv_summary[indiv_summary.index.str.contains('alpha\\[', regex=True)])
    pgain_count = len(indiv_summary[indiv_summary.index.str.contains('p_gain_effect\\[', regex=True)])
    rthreat_count = len(indiv_summary[indiv_summary.index.str.contains('r_threat_effect\\[', regex=True)])
    
    print(f"Alpha (intercepts): {alpha_count} participants")
    print(f"P_gain effects: {pgain_count} participants")
    print(f"R_threat effects: {rthreat_count} participants")
    print(f"\nMax R_hat across all individual parameters: {indiv_summary['r_hat'].max():.4f}")
    if indiv_summary['r_hat'].max() < 1.01:
        print("✓ All individual-level parameters well-converged")
    else:
        print("⚠ Some individual-level parameters have high R_hat values")

print("\n✓ Model stored as: hierarchical_model")
print("✓ Posterior samples stored as: idata_pgain")

