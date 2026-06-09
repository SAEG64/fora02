# HIERARCHICAL MIXTURE OF POLICIES WITH CONDITION + BLOCK EFFECTS
#
##
###
# %% Data Preprocessing
""" Prepare data for hierarchical mixture model with condition and block effects. """
# ==============================================================================

import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
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

print(f"\nData cleaning: Removed {len(df.index)} - Valid trials for modeling")

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
# CONDITION CODING
# ------------------------------------------------------------------------------
cond = 1-pd.factorize(df["condition_rORp"])[0]
cond_signed = 2 * cond - 1   # -1 / +1

subj_idx = df["subj_idx"].values.astype(int)

# %% Get Model Predictions
""" FIT HIERARCHICAL OP + MH MODELS FOR POLICY ARBITRATION """
# ==============================================================================

print("\n" + "="*80)
print("FITTING OP + MH POLICY MODELS FOR ARBITRATION ANALYSIS")
print("="*80)

# ------------------------------------------------------------------------------
# PREPARE DATA
# ------------------------------------------------------------------------------

# Response
y_fit = df["foraging T/F NaNs"].values.astype(int)

# Subject index
subj_idx_fit = df["subj_idx"].values.astype(int)

# Number of subjects
n_subjects_fit = len(np.unique(subj_idx_fit))

# ------------------------------------------------------------------------------
# OP MODEL DATA
# ------------------------------------------------------------------------------

X_op = df["OP_value_difference"].values.astype(float)

# Standardize OP predictor
X_op = (X_op - X_op.mean()) / X_op.std()

# ------------------------------------------------------------------------------
# MH MODEL DATA
# ------------------------------------------------------------------------------

X_mh = np.column_stack([

    # p(gain)
    (
        df["* $\\mathit{p}$ gain"].values.astype(float)
        - df["* $\\mathit{p}$ gain"].mean()
    )
    / df["* $\\mathit{p}$ gain"].std(),

    # r(predator)
    (
        df["* $\\mathit{r}$ predator"].values.astype(float)
        - df["* $\\mathit{r}$ predator"].mean()
    )
    / df["* $\\mathit{r}$ predator"].std(),

    # WWS
    df["** wait when safe"].values.astype(float),

    # BES
    df["** binary energy state"].values.astype(float)

])

# ------------------------------------------------------------------------------
# MH INTERACTIONS
# ------------------------------------------------------------------------------

interactions_mh = np.column_stack([

    # p_gain × predator
    X_mh[:, 0] * X_mh[:, 1],

    # p_gain × WWS
    X_mh[:, 0] * X_mh[:, 2],

    # p_gain × BES
    X_mh[:, 0] * X_mh[:, 3],

    # predator × WWS
    X_mh[:, 1] * X_mh[:, 2],

    # predator × BES
    X_mh[:, 1] * X_mh[:, 3]

])

# Standardization
interactions_mh = (
    interactions_mh
    - interactions_mh.mean(axis=0)
) / interactions_mh.std(axis=0)

# ==============================================================================
# 1. FIT OP MODEL
# ==============================================================================

with pm.Model() as op_model:

    # --------------------------------------------------------------------------
    # HIERARCHICAL INTERCEPT
    # --------------------------------------------------------------------------

    mu_alpha_op = pm.Normal("mu_alpha_op", 0, 1)
    sigma_alpha_op = pm.HalfNormal("sigma_alpha_op", 1)

    alpha_offset_op = pm.Normal(
        "alpha_offset_op",
        0,
        1,
        shape=n_subjects_fit
    )

    alpha_op = (
        mu_alpha_op
        + sigma_alpha_op * alpha_offset_op
    )

    # --------------------------------------------------------------------------
    # HIERARCHICAL SLOPE
    # --------------------------------------------------------------------------

    mu_beta_op = pm.Normal("mu_beta_op", 0, 1)
    sigma_beta_op = pm.HalfNormal("sigma_beta_op", 1)

    beta_offset_op = pm.Normal(
        "beta_offset_op",
        0,
        1,
        shape=n_subjects_fit
    )

    beta_op = (
        mu_beta_op
        + sigma_beta_op * beta_offset_op
    )

    # --------------------------------------------------------------------------
    # LINEAR PREDICTOR
    # --------------------------------------------------------------------------

    eta_op = pm.Deterministic(
        "eta_op",
        alpha_op[subj_idx_fit]
        + beta_op[subj_idx_fit] * X_op
    )

    # --------------------------------------------------------------------------
    # LIKELIHOOD
    # --------------------------------------------------------------------------

    y_obs_op = pm.Bernoulli(
        "y_obs_op",
        logit_p=eta_op,
        observed=y_fit
    )

    # --------------------------------------------------------------------------
    # SAMPLING
    # --------------------------------------------------------------------------

    idata_op = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=0.9,
        random_seed=42
    )

# ==============================================================================
# 2. FIT MH MODEL
# ==============================================================================

with pm.Model() as mh_model:

    # --------------------------------------------------------------------------
    # HIERARCHICAL INTERCEPT
    # --------------------------------------------------------------------------

    mu_alpha_mh = pm.Normal("mu_alpha_mh", 0, 1)
    sigma_alpha_mh = pm.HalfNormal("sigma_alpha_mh", 1)

    alpha_offset_mh = pm.Normal(
        "alpha_offset_mh",
        0,
        1,
        shape=n_subjects_fit
    )

    alpha_mh = (
        mu_alpha_mh
        + sigma_alpha_mh * alpha_offset_mh
    )

    # --------------------------------------------------------------------------
    # HIERARCHICAL FEATURE SLOPES
    # --------------------------------------------------------------------------

    n_features_mh = X_mh.shape[1]

    mu_beta_mh = pm.Normal(
        "mu_beta_mh",
        0,
        1,
        shape=n_features_mh
    )

    sigma_beta_mh = pm.HalfNormal(
        "sigma_beta_mh",
        1,
        shape=n_features_mh
    )

    beta_offset_mh = pm.Normal(
        "beta_offset_mh",
        0,
        1,
        shape=(n_subjects_fit, n_features_mh)
    )

    beta_mh = (
        mu_beta_mh[np.newaxis, :]
        + sigma_beta_mh[np.newaxis, :] * beta_offset_mh
    )

    # --------------------------------------------------------------------------
    # FIXED INTERACTION TERMS
    # --------------------------------------------------------------------------

    n_interactions_mh = interactions_mh.shape[1]

    beta_interactions_mh = pm.Normal(
        "beta_interactions_mh",
        0,
        1,
        shape=n_interactions_mh
    )

    # --------------------------------------------------------------------------
    # LINEAR PREDICTOR
    # --------------------------------------------------------------------------

    eta_mh = pm.Deterministic(
        "eta_mh",

        alpha_mh[subj_idx_fit]

        + pm.math.sum(
            beta_mh[subj_idx_fit] * X_mh,
            axis=1
        )

        + pm.math.sum(
            beta_interactions_mh * interactions_mh,
            axis=1
        )
    )

    # --------------------------------------------------------------------------
    # LIKELIHOOD
    # --------------------------------------------------------------------------

    y_obs_mh = pm.Bernoulli(
        "y_obs_mh",
        logit_p=eta_mh,
        observed=y_fit
    )

    # --------------------------------------------------------------------------
    # SAMPLING
    # --------------------------------------------------------------------------

    idata_mh = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=0.9,
        random_seed=42
    )

# ==============================================================================
# EXTRACT POSTERIOR POLICY PREDICTIONS
# ==============================================================================

print("\nExtracting posterior policy predictions...")

# ------------------------------------------------------------------------------
# POSTERIOR MEAN LOGITS
# ------------------------------------------------------------------------------

eta_op_mean = (
    idata_op.posterior["eta_op"]
    .mean(dim=("chain", "draw"))
    .values
)

eta_mh_mean = (
    idata_mh.posterior["eta_mh"]
    .mean(dim=("chain", "draw"))
    .values
)

# ------------------------------------------------------------------------------
# CONVERT TO PROBABILITIES
# ------------------------------------------------------------------------------

p_op_model = 1 / (1 + np.exp(-eta_op_mean))
p_mh_model = 1 / (1 + np.exp(-eta_mh_mean))

# ------------------------------------------------------------------------------
# ADD TO DATAFRAME
# ------------------------------------------------------------------------------

df["p_op_model"] = p_op_model
df["p_mh_model"] = p_mh_model

print("Done.")
print("Added:")
print("  - p_op_model")
print("  - p_mh_model")

# %% Preprocessing Policy Predictors
""" PREPARE POLICY PREDICTIONS FOR ARBITRATION MODEL """
# ==============================================================================

print("\nPreparing policy predictions and responses for arbitration model...")

# ------------------------------------------------------------------------------
# SAFETY CLIPPING
# ------------------------------------------------------------------------------

eps = 1e-6

p_op_model = np.clip(p_op_model, eps, 1 - eps)
p_mh_model = np.clip(p_mh_model, eps, 1 - eps)

# ------------------------------------------------------------------------------
# CONVERT TO LOGITS
# ------------------------------------------------------------------------------

logit_p_op = np.log(
    p_op_model / (1 - p_op_model)
)

logit_p_mh = np.log(
    p_mh_model / (1 - p_mh_model)
)

# ------------------------------------------------------------------------------
# POLICY DISAGREEMENT (delta_t)
# ------------------------------------------------------------------------------

delta_t = np.abs(
    p_op_model - p_mh_model
)

# Standardize delta_t
delta_t = (
    delta_t - delta_t.mean()
) / delta_t.std()

# ------------------------------------------------------------------------------
# STANDARDIZE BLOCK
# ------------------------------------------------------------------------------

block_z = (
    df["block_idx"].values.astype(float)
    - df["block_idx"].mean()
) / df["block_idx"].std()

# ------------------------------------------------------------------------------
# CONDITION CODING
# ------------------------------------------------------------------------------

# avoidance = +1
# approach = -1

cond = 1 - pd.factorize(df["condition_rORp"])[0]
cond_signed = 2 * cond - 1

# ------------------------------------------------------------------------------
# SUBJECT INDEX
# ------------------------------------------------------------------------------

subj_idx = df["subj_idx"].values.astype(int)
n_subj = len(np.unique(subj_idx))

print("Done.")

# ==============================================================================
# 2. MODEL SETUP
# ==============================================================================

# ------------------------------------------------------------------------------
# RESPONSE VARIABLE
# ------------------------------------------------------------------------------

y = df["foraging T/F NaNs"].values.astype(int)

# delta_t already calculated and standardized above
delta_t_model = delta_t.copy()

# ==============================================================================
# 2. MODEL SEPARABILITY INDEX (OP vs MH MISALIGNMENT)
# ==============================================================================

print("\n" + "="*80)
print("MODEL SEPARABILITY ANALYSIS: OP vs MH MISALIGNMENT")
print("="*80)

# delta_t already calculated above; store in df for analysis
df["delta_t"] = delta_t

# Identify misaligned trials (delta_t > 0.1, indicating meaningful disagreement)
misalignment_threshold = 0.1
df["is_misaligned"] = df["delta_t"] > misalignment_threshold

# Filter to misaligned trials only
df_misaligned = df[df["is_misaligned"]].copy()

print(f"\nTotal trials: {len(df)}")
print(f"Misaligned trials (delta_t > {misalignment_threshold}): {len(df_misaligned)} ({100*len(df_misaligned)/len(df):.1f}%)")

# Factorize condition for misaligned data
df_misaligned['condition_factor'], condition_labels = pd.factorize(df_misaligned["condition_rORp"])

# Create condition label mapping for printing
condition_rename_map = {
    "high threat condition": "avoidance",
    "low threat condition": "approach"
}

# Descriptive statistics for delta_t
print("\n--- DESCRIPTIVE STATISTICS FOR DELTA_T (Model Separability) ---")
print("\nBy Condition:")
condition_stats = df_misaligned.groupby("condition_rORp")['delta_t'].agg(['count', 'mean', 'std', 'sem']).round(4)
condition_stats.index = condition_stats.index.map(lambda x: condition_rename_map.get(x, x))
print(condition_stats)

print("\nBy Block:")
print(df_misaligned.groupby("block_idx")['delta_t'].agg(['count', 'mean', 'std', 'sem']).round(4))

print("\nBy Condition and Block:")
summary_sep = df_misaligned.groupby(['condition_rORp', 'block_idx'])['delta_t'].agg([
    'count', 'mean', 'std', 'sem'
]).round(4)
summary_sep.index = summary_sep.index.set_levels(
    summary_sep.index.levels[0].map(lambda x: condition_rename_map.get(x, x)), 
    level=0
)
print(summary_sep)

# ==============================================================================
# 2.1 PUBLICATION-QUALITY VISUALIZATIONS FOR MODEL SEPARABILITY
# ==============================================================================

import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 600

# Prepare aggregated data
sep_summary = df_misaligned.groupby('condition_rORp').agg({
    'delta_t': ['mean', 'sem', 'count']
}).reset_index()
sep_summary.columns = ['condition', 'delta_t_mean', 'delta_t_sem', 'n']

# Map condition to labels
condition_map = {condition_labels[0]: 'Avoidance', condition_labels[1]: 'Approach'}
sep_summary['condition_label'] = sep_summary['condition'].map(condition_map)

# ============================================================================
# Figure 1: Mean Model Separability by Condition
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5.5))

colors = ['#E74C3C', '#3498DB']
bars = ax.bar(sep_summary['condition_label'], sep_summary['delta_t_mean'],
              yerr=sep_summary['delta_t_sem'], capsize=8, width=0.5,
              color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Mean Model Separability (|ΔP|)', fontsize=12, fontweight='bold')
ax.set_xlabel('Forest Condition', fontsize=12, fontweight='bold')
ax.set_title('Model Separability Index by Condition\n(OP vs MH Misalignment)', 
             fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add sample sizes on bars
for i, (bar, n) in enumerate(zip(bars, sep_summary['n'])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'n={int(n)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(path + 'RESULTS/delta_t_by_condition_barplot.png', dpi=600, bbox_inches='tight')
plt.show()

# ============================================================================
# Figure 2: Distribution of Model Separability by Condition (Histograms)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for idx, (condition, cond_label) in enumerate(zip([condition_labels[0], condition_labels[1]], 
                                                    ['Avoidance', 'Approach'])):
    ax = axes[idx]
    cond_data = df_misaligned[df_misaligned['condition_rORp'] == condition]['delta_t']
    color = '#E74C3C' if idx == 0 else '#3498DB'
    
    # Histogram with KDE
    ax.hist(cond_data, bins=30, color=color, alpha=0.6, edgecolor='black', 
            linewidth=1.2, density=True, label='Histogram')
    
    # Add KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(cond_data)
    x_range = np.linspace(cond_data.min(), cond_data.max(), 200)
    ax.plot(x_range, kde(x_range), color='black', linewidth=2.5, label='KDE')
    
    # Add vertical line for mean
    ax.axvline(cond_data.mean(), color='darkred' if idx == 0 else 'darkblue', 
               linestyle='--', linewidth=2.5, label=f'Mean = {cond_data.mean():.3f}')
    
    ax.set_xlabel('Model Separability (|ΔP|)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title(f'{cond_label} Forests (n={len(cond_data)})', 
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Distribution of Model Separability by Forest Condition', 
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(path + 'RESULTS/delta_t_distributions_histograms.png', dpi=600, bbox_inches='tight')
plt.show()

# ============================================================================
# Figure 3: Violin Plots with Individual Points
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

data_for_plot = []
labels_for_plot = []

for condition, cond_label in zip([condition_labels[0], condition_labels[1]], 
                                  ['Avoidance', 'Approach']):
    cond_data = df_misaligned[df_misaligned['condition_rORp'] == condition]['delta_t'].values
    data_for_plot.append(cond_data)
    labels_for_plot.append(cond_label)

# Create violin plot
parts = ax.violinplot(data_for_plot, positions=[0, 1], widths=0.7,
                      showmeans=True, showmedians=True)

# Customize colors
colors = ['#E74C3C', '#3498DB']
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')

# Add individual points with jitter
for i, (data, color) in enumerate(zip(data_for_plot, colors)):
    yy = data
    xx = np.random.normal(i, 0.04, size=len(yy))
    ax.scatter(xx, yy, alpha=0.25, s=25, color=color, edgecolor='none')

ax.set_ylabel('Model Separability (|ΔP|)', fontsize=12, fontweight='bold')
ax.set_xlabel('Forest Condition', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Model Separability: Individual Trials', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks([0, 1])
ax.set_xticklabels(labels_for_plot)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(path + 'RESULTS/delta_t_violin_plots.png', dpi=600, bbox_inches='tight')
plt.show()

# ============================================================================
# Figure 4: Model Separability Across Blocks by Condition
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

sep_by_block = df_misaligned.groupby(['block_idx', 'condition_rORp'])['delta_t'].agg(['mean', 'sem']).reset_index()
sep_by_block['condition_label'] = sep_by_block['condition_rORp'].map(condition_map)

for cond_label, color in zip(['Avoidance', 'Approach'], ['#E74C3C', '#3498DB']):
    cond_data = sep_by_block[sep_by_block['condition_label'] == cond_label]
    ax.errorbar(cond_data['block_idx'], cond_data['mean'],
                yerr=cond_data['sem'], 
                marker='o', markersize=7, capsize=5, capthick=1.5,
                linewidth=2.5, label=cond_label, color=color, alpha=0.85)

ax.set_xlabel('Block Number', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Model Separability (|ΔP|)', fontsize=12, fontweight='bold')
ax.set_title('Model Separability Across Blocks by Condition', 
             fontsize=14, fontweight='bold', pad=15)
ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='best', title='Condition')
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(path + 'RESULTS/delta_t_across_blocks.png', dpi=600, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("MODEL SEPARABILITY VISUALIZATIONS SAVED TO RESULTS/")
print("="*80)
print("✓ delta_t_by_condition_barplot.png - Mean separability by condition")
print("✓ delta_t_distributions_histograms.png - Distribution histograms with KDE")
print("✓ delta_t_violin_plots.png - Violin plots with individual points")
print("✓ delta_t_across_blocks.png - Separability trends across blocks")

# %% Arbitration Analysis
""" OP vs MH mixture with hierarchical arbitration
    including condition, learning (block), and
    policy separability (delta_t) as covariates.

Model Interpretation:
- π = P(use MH) is modeled as a function of condition, block, and delta_t
- delta_t is the trial-level model separability index |P(OP) - P(MH)|
- Positive beta_delta_t: Higher OP-MH misalignment → increased use of MH (relative to OP)
- Negative beta_delta_t: Higher OP-MH misalignment → increased use of OP (relative to MH)
- This tests whether people adaptively choose models based on their disagreement
"""
# ==============================================================================
# 2. MODEL SETUP (Extract delta_t for model)
# ==============================================================================

# delta_t is already calculated and standardized above, ready for the model
delta_t_model = delta_t.copy()

# ==============================================================================
# 2.1 HIERARCHICAL MIXTURE MODEL
# ==============================================================================

with pm.Model() as mix_model:

    # --------------------------------------------------------------------------
    # 2.2 MIXTURE (π) WITH CONDITION + BLOCK
    # --------------------------------------------------------------------------

    # Baseline
    mu_logit_pi = pm.Normal("mu_logit_pi", 0, 1.0)
    sigma_logit_pi = pm.HalfNormal("sigma_logit_pi", 1.0)

    logit_pi_offset = pm.Normal("logit_pi_offset", 0, 1, shape=n_subj)

    logit_pi_subj = mu_logit_pi + logit_pi_offset * sigma_logit_pi
    logit_pi_subj_t = logit_pi_subj[subj_idx]

    # Condition effect
    mu_beta_cond_pi = pm.Normal("mu_beta_cond_pi", 0, 1.0)
    sigma_beta_cond_pi = pm.HalfNormal("sigma_beta_cond_pi", 1.0)

    beta_cond_offset = pm.Normal("beta_cond_offset", 0, 1, shape=n_subj)

    beta_cond_subj = pm.Deterministic(
        "beta_cond_subj",
        mu_beta_cond_pi + beta_cond_offset * sigma_beta_cond_pi
    )

    beta_cond_t = beta_cond_subj[subj_idx]

    # Block effect (learning)
    mu_beta_block_pi = pm.Normal("mu_beta_block_pi", 0, 1.0)
    sigma_beta_block_pi = pm.HalfNormal("sigma_beta_block_pi", 1.0)

    beta_block_offset = pm.Normal("beta_block_offset", 0, 1, shape=n_subj)

    beta_block_subj = pm.Deterministic(
        "beta_block_subj",
        mu_beta_block_pi + beta_block_offset * sigma_beta_block_pi
    )

    beta_block_t = beta_block_subj[subj_idx]

    # Interaction
    mu_beta_interact = pm.Normal("mu_beta_interact", 0, 1.0)

    # Delta_t effect (policy separability/misalignment)
    mu_beta_delta_t = pm.Normal("mu_beta_delta_t", 0, 1.0)
    sigma_beta_delta_t = pm.HalfNormal("sigma_beta_delta_t", 1.0)

    beta_delta_t_offset = pm.Normal("beta_delta_t_offset", 0, 1, shape=n_subj)

    beta_delta_t_subj = pm.Deterministic(
        "beta_delta_t_subj",
        mu_beta_delta_t + beta_delta_t_offset * sigma_beta_delta_t
    )

    beta_delta_t_t = beta_delta_t_subj[subj_idx]

    # Final mixture logit with delta_t covariate
    delta_t_data = pm.Data("delta_t_data", delta_t_model)
    
    logit_pi = (
        logit_pi_subj_t
        + beta_cond_t * cond_signed
        + beta_block_t * block_z
        + beta_delta_t_t * delta_t_data
        + mu_beta_interact * cond_signed * block_z
    )

    pi = pm.Deterministic("pi", pm.math.sigmoid(logit_pi))

    # --------------------------------------------------------------------------
    # 2.4 MIXTURE
    # --------------------------------------------------------------------------
    # log probabilities
    log_p_op = -pt.math.softplus(-logit_p_op)
    log_p_mh = -pt.math.softplus(-logit_p_mh)

    log1m_p_op = -pt.math.softplus(logit_p_op)
    log1m_p_mh = -pt.math.softplus(logit_p_mh)

    # log mixture for y = 1
    log_mix_p = pt.math.logaddexp(
        pt.log(pi) + log_p_mh,
        pt.log(1 - pi) + log_p_op
    )

    # log mixture for y = 0
    log_mix_1mp = pt.math.logaddexp(
        pt.log(pi) + log1m_p_mh,
        pt.log(1 - pi) + log1m_p_op
    )

    # full log-likelihood
    logp = y * log_mix_p + (1 - y) * log_mix_1mp

    # register likelihood
    pm.Potential("likelihood", logp)

    # ==============================================================================
    # 3. SAMPLING
    # ==============================================================================

    idata_mix = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=0.97,
        random_seed=42
    )

# ==============================================================================
# 4. ANALYSIS
# ==============================================================================

print("\n=== MODEL SUMMARY ===")
print(az.summary(idata_mix, var_names=[
    "mu_logit_pi",
    "mu_beta_cond_pi",
    "mu_beta_block_pi",
    "mu_beta_interact"
]))

# ------------------------------------------------------------------------------
# 4.1 SAFE π EXTRACTION
# ------------------------------------------------------------------------------
pi_post = idata_mix.posterior["pi"].mean(dim=("chain", "draw")).values

if pi_post.ndim > 1:
    pi_post = pi_post.reshape(-1)

df = df.copy()
df["pi"] = pi_post

# ------------------------------------------------------------------------------
# 4.2 BLOCK-LEVEL EFFECT
# ------------------------------------------------------------------------------
df_block = df.groupby("block_idx")["pi"].mean().reset_index()

print("\n=== BLOCK EFFECT (π) ===")
print(df_block)

# ------------------------------------------------------------------------------
# 4.3 SUBJECT-LEVEL LEARNING
# ------------------------------------------------------------------------------
beta_block = idata_mix.posterior["beta_block_subj"].mean(dim=("chain","draw")).values

print("\n=== LEARNING EFFECT ===")
print("Mean:", beta_block.mean())
print("Subjects shifting to OP:", np.sum(beta_block < 0))

# ------------------------------------------------------------------------------
# 4.4 CONDITION EFFECT
# ------------------------------------------------------------------------------
beta_cond = idata_mix.posterior["beta_cond_subj"].mean(dim=("chain","draw")).values

print("\n=== CONDITION EFFECT ===")
print("Mean:", beta_cond.mean())

# ------------------------------------------------------------------------------
# 4.5 GROUP-LEVEL LEARNING
# ------------------------------------------------------------------------------
mu_block = idata_mix.posterior["mu_beta_block_pi"].values.flatten()

print("\n=== GROUP LEARNING ===")
print("Mean:", mu_block.mean())
print("P(OP increase):", (mu_block < 0).mean())

beta_delta = idata_mix.posterior["mu_beta_delta_t"].values.flatten()

print("\n=== DELTA_T EFFECT (SYMMETRY / SEPARABILITY) ===")
print("Mean:", beta_delta.mean())
print("P(>0):", (beta_delta > 0).mean())
print("P(<0):", (beta_delta < 0).mean())

beta_delta_subj = idata_mix.posterior["beta_delta_t_subj"].mean(dim=("chain","draw")).values

print("Subjects with positive effect:", np.sum(beta_delta_subj > 0))
print("Subjects with negative effect:", np.sum(beta_delta_subj < 0))

# %% CONDITION EFFECT — SUBJECT HETEROGENEITY
print("\n=== CONDITION EFFECT: SUBJECT HETEROGENEITY ===")

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# 1. EXTRACT POSTERIOR SAMPLES
# shape: (chains, draws, subjects)
# --------------------------------------------------------------------------
beta_samples = idata_mix.posterior["beta_cond_subj"].values

# --------------------------------------------------------------------------
# 2. PROBABILITY OF SHIFT TOWARD OP
# beta < 0 → shift toward OP
# --------------------------------------------------------------------------
p_op_shift = (beta_samples < 0).mean(axis=(0, 1))

print("Number of subjects:", len(p_op_shift))
print("Range:", f"{p_op_shift.min():.2f}", "to", f"{p_op_shift.max():.2f}")

# --------------------------------------------------------------------------
# 3. PLOT HISTOGRAM (PUBLICATION-READY)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

# Histogram with professional styling
ax.hist(p_op_shift, bins=12, color="#2E86AB", edgecolor="white", 
        linewidth=1.5, alpha=0.85)

# Reference lines with distinct styling
ax.axvline(p_op_shift.mean(), color="#A23B72", linestyle="-", linewidth=2.5, 
           label=f"Mean: {p_op_shift.mean():.2f}", alpha=0.9)
ax.axvline(0.1, color="#C73E1D", linestyle=":", linewidth=2.5, 
           label="p = 0.10", alpha=0.8)
ax.axvline(0.5, color="#F18F01", linestyle="--", linewidth=2, 
           label="p = 0.50", alpha=0.8)
ax.axvline(0.9, color="#C73E1D", linestyle=":", linewidth=2.5, 
           label="p = 0.90", alpha=0.8)

# Professional labels
ax.set_xlabel(r"P(shift toward $\Delta \mathit{Q}$ | avoidance)", fontsize=18)
ax.set_ylabel("Number of subjects", fontsize=18)
ax.set_title("Condition-Dependent Policy Shift", 
             fontsize=22, pad=15)

# Grid and styling
ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)

# Font sizes for ticks
ax.tick_params(axis="both", which="major", labelsize=11)

# Legend
ax.legend(fontsize=14, loc="upper left", framealpha=0.95, edgecolor="black")

plt.tight_layout()
plt.show()

# %% BASELINE POLICY USE (OP vs MH)
print("\n=== BASELINE POLICY USE ===")

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd

# --------------------------------------------------------------------------
# 1. GROUP-LEVEL BASELINE π
# --------------------------------------------------------------------------
summary_main = az.summary(idata_mix, var_names=["mu_logit_pi", "sigma_logit_pi"])
print(summary_main)

# Convert logit → probability
mu_logit_samples = idata_mix.posterior["mu_logit_pi"].values
pi_group = 1 / (1 + np.exp(-mu_logit_samples))

pi_group_flat = pi_group.flatten()
pi_group_mean = pi_group_flat.mean()

hdi = az.hdi(pi_group_flat, hdi_prob=0.95)
hdi_low, hdi_high = float(hdi[0]), float(hdi[1])

print("\nGroup-level baseline π (P(MH)):")
print(f"Mean: {pi_group_mean:.3f}")
print(f"95% HDI: [{hdi_low:.3f}, {hdi_high:.3f}]")

if pi_group_mean > 0.5:
    print("→ Bias toward MH")
else:
    print("→ Bias toward OP")

# --------------------------------------------------------------------------
# 2. SUBJECT-LEVEL BASELINE POLICY USE
# --------------------------------------------------------------------------
logit_pi_offset = idata_mix.posterior["logit_pi_offset"].values
mu = idata_mix.posterior["mu_logit_pi"].values
sigma = idata_mix.posterior["sigma_logit_pi"].values

logit_pi_subj = mu[..., None] + logit_pi_offset * sigma[..., None]
pi_subj = 1 / (1 + np.exp(-logit_pi_subj))

pi_subj_mean = pi_subj.mean(axis=(0,1))

# --------------------------------------------------------------------------
# 3. PLOT DISTRIBUTION (PUBLICATION-READY)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

# Histogram with professional styling
ax.hist(pi_subj_mean, bins=12, color="#06A77D", edgecolor="white", 
        linewidth=1.5, alpha=0.85)

# Reference lines with statistical information
ax.axvline(pi_subj_mean.mean(), color="#D62246", linestyle="-", linewidth=2.5, 
           label=f"Mean: {pi_subj_mean.mean():.3f}", alpha=0.9)
ax.axvline(np.median(pi_subj_mean), color="#1B998B", linestyle="--", linewidth=2, 
           label=f"Median: {np.median(pi_subj_mean):.3f}", alpha=0.8)
ax.axvline(0.5, color="#F18F01", linestyle=":", linewidth=2.5, 
           label="Indifference", alpha=0.8)

# Professional labels
ax.set_xlabel("P(multi-feature policy)", fontsize=18)
ax.set_ylabel("Number of subjects", fontsize=18)
ax.set_title("Mixture Weights", 
             fontsize=22, pad=15)

# Grid and styling
ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)

# Font sizes for ticks
ax.tick_params(axis="both", which="major", labelsize=11)

# Legend
ax.legend(fontsize=14, loc="center left", framealpha=0.95, edgecolor="black")

# Statistical annotation
stats_text = f"n = {len(pi_subj_mean)}\nSD = {pi_subj_mean.std():.3f}"
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------
# 4. HETEROGENEITY TEST
# --------------------------------------------------------------------------
sigma_samples = idata_mix.posterior["sigma_logit_pi"].values.flatten()

print("\n=== HETEROGENEITY TEST ===")
print(f"Mean sigma_logit_pi: {sigma_samples.mean():.3f}")
print(f"P(sigma > 0.1): {(sigma_samples > 0.1).mean():.3f}")

# --------------------------------------------------------------------------
# 5. SUBJECT TABLE
# --------------------------------------------------------------------------
df_policy = pd.DataFrame({
    "subj": subj_ids,
    "pi_baseline": pi_subj_mean
})

print("\n=== SUBJECT-LEVEL BASELINE POLICY USE ===")
print(df_policy.head())

# %% CONDITION EFFECT: DIFFERENTIAL OP ADOPTION
print("\n=== CONDITION EFFECT: DIFFERENTIAL OP ADOPTION ===")

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# 1. EXTRACT POSTERIOR SAMPLES
# --------------------------------------------------------------------------
logit_base = idata_mix.posterior["logit_pi_offset"].values
mu = idata_mix.posterior["mu_logit_pi"].values
sigma = idata_mix.posterior["sigma_logit_pi"].values

beta_cond = idata_mix.posterior["beta_cond_subj"].values

# --------------------------------------------------------------------------
# 2. RECONSTRUCT SUBJECT BASELINE AND CONDITION-SPECIFIC π
# --------------------------------------------------------------------------
logit_subj = mu[..., None] + logit_base * sigma[..., None]

pi_avoid = 1 / (1 + np.exp(-(logit_subj + beta_cond)))
pi_approach = 1 / (1 + np.exp(-(logit_subj - beta_cond)))

# Compute differential OP adoption: ΔP(OP) = P(OP|approach) - P(OP|avoid)
delta_op_subj = (1 - pi_avoid) - (1 - pi_approach)

delta_mean = delta_op_subj.mean(axis=(0, 1))

print("Number of subjects:", len(delta_mean))
print("Mean ΔP(OP):", f"{delta_mean.mean():.2f}")
print("Range:", f"{delta_mean.min():.2f}", "to", f"{delta_mean.max():.2f}")

# --------------------------------------------------------------------------
# 3. PLOT HISTOGRAM (PUBLICATION-READY)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

# Histogram with professional styling
ax.hist(delta_mean, bins=12, color="#E63946", edgecolor="white", 
        linewidth=1.5, alpha=0.85)

# Reference lines
ax.axvline(delta_mean.mean(), color="#2E86AB", linestyle="-", linewidth=2.5, 
           label=f"Mean: {delta_mean.mean():.2f}", alpha=0.9)
ax.axvline(0, color="#F18F01", linestyle="--", linewidth=2, 
           label="No effect (Δ = 0)", alpha=0.8)

# Professional labels
ax.set_xlabel(r"$\Delta P(\Delta \mathit{Q})$ [Approach − Avoidance]", fontsize=18)
ax.set_ylabel("Number of subjects", fontsize=18)
ax.set_title("Condition-Dependent Shift", 
             fontsize=22, pad=15)

# Grid and styling
ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)

# Font sizes for ticks
ax.tick_params(axis="both", which="major", labelsize=11)

# Legend
ax.legend(fontsize=12, loc="upper right", framealpha=0.95, edgecolor="black")

# Statistical annotation
stats_text = f"n = {len(delta_mean)}\nSD = {delta_mean.std():.2f}"
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.tight_layout()
plt.show()