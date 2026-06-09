
# %% Preprocessing
""" Get data ready for analysis """
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
# BLOCK CREATION (PRESERVE TRUE ORDER)
# ------------------------------------------------------------------------------

# Previous miniblock value per subject
prev_mb = df.groupby("Subject_ID")["miniblock.thisN"].shift()

# Detect FIRST 0 of each sequence
df["block_start"] = (
    (df["miniblock.thisN"] == 0) & (prev_mb != 0)
).astype(int)

# Count blocks per subject
df["block_idx"] = df.groupby("Subject_ID")["block_start"].cumsum() -1

# %% RT, Blocks, and Conditions Effects
# 2. ANALYSIS OF logRT PER SUBJECT, BLOCK, AND CONDITION
# ==============================================================================

import statsmodels.formula.api as smf

# Prepare data for analysis
df_logrt = df[['Subject_ID', 'subj_idx', 'block_idx', 'condition_rORp', 'logRT']].dropna()

print("\n" + "="*80)
print("LogRT ANALYSIS: BLOCKS, CONDITIONS, AND INTERACTION")
print("="*80)

# Factorize condition first, then convert to integer
df_logrt['condition_rORp'], condition_labels = pd.factorize(df_logrt['condition_rORp'])

# Descriptive statistics grouped by condition and block
print("\n--- DESCRIPTIVE STATISTICS BY CONDITION AND BLOCK ---")
summary = df_logrt.groupby(['condition_rORp', 'block_idx'])['logRT'].agg([
    'count', 'mean', 'std', 'sem'
]).round(4)
print(summary)

# Overall descriptive statistics by condition
print("\n--- DESCRIPTIVE STATISTICS BY CONDITION ---")
summary_cond = df_logrt.groupby('condition_rORp')['logRT'].agg([
    'count', 'mean', 'std', 'sem'
]).round(4)
print(summary_cond)

# Overall descriptive statistics by block
print("\n--- DESCRIPTIVE STATISTICS BY BLOCK ---")
summary_block = df_logrt.groupby('block_idx')['logRT'].agg([
    'count', 'mean', 'std', 'sem'
]).round(4)
print(summary_block)

# Mixed-effects model: logRT ~ block + condition + block:condition + (1|Subject_ID)
print("\n--- MIXED-EFFECTS MODEL ---")
print("Formula: logRT ~ C(block_idx) + C(condition_rORp) + C(block_idx):C(condition_rORp) + (1|Subject_ID)")

model = smf.mixedlm(
    "logRT ~ C(block_idx) + C(condition_rORp) + C(block_idx):C(condition_rORp)",
    df_logrt,
    groups=df_logrt["Subject_ID"]
)
result = model.fit()
print(result.summary())

# Extract and display fixed effects
print("\n--- FIXED EFFECTS (Parameter Estimates) ---")
print(result.fe_params.round(4))

# P-values for fixed effects
print("\n--- P-VALUES FOR FIXED EFFECTS ---")
print(result.pvalues.round(4))

# Extract coefficients with confidence intervals
print("\n--- FIXED EFFECTS WITH 95% CONFIDENCE INTERVALS ---")
conf_int = result.conf_int(
)
conf_int.columns = ['Lower CI', 'Upper CI']
params_ci = pd.concat([result.fe_params.rename('Estimate'), conf_int], axis=1)
print(params_ci.round(4))

# Test for main effect of block (omnibus F-test)
print("\n--- OMNIBUS F-TESTS FOR MAIN EFFECTS ---")

# Create design matrices for testing
from sklearn.preprocessing import LabelEncoder

# Fit full model (already done above)
n_obs = len(df_logrt)
n_groups = df_logrt['Subject_ID'].nunique()

# Get unique blocks and conditions
n_blocks = df_logrt['block_idx'].nunique()
n_conditions = df_logrt['condition_rORp'].nunique()

print(f"Number of observations: {n_obs}")
print(f"Number of subjects: {n_groups}")
print(f"Number of blocks: {n_blocks}")
print(f"Number of conditions: {n_conditions}")

# Calculate R-squared for model fit
print(f"\nModel fit:")
print(f"  AIC: {result.aic:.2f}")
print(f"  BIC: {result.bic:.2f}")

# Visualization: Mean logRT by block and condition
print("\n--- GENERATING SUMMARY TABLE FOR VISUALIZATION ---")
fig_data = df_logrt.groupby(['block_idx', 'condition_rORp'])['logRT'].agg(['mean', 'std', 'sem', 'count'])
print(fig_data.round(4))

# ==============================================================================
# 3. PUBLICATION-QUALITY VISUALIZATIONS
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
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 6

# Prepare data for plotting
plot_data = df_logrt.groupby(['block_idx', 'condition_rORp']).agg({
    'logRT': ['mean', 'sem']
}).reset_index()
plot_data.columns = ['block_idx', 'condition_rORp', 'logRT_mean', 'logRT_sem']
plot_data['condition_label'] = plot_data['condition_rORp'].map({0: 'Avoidance', 1: 'Approach'})

# ============================================================================
# Figure 1: Line plot - Mean logRT across blocks by condition
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

for condition in [0, 1]:
    cond_data = plot_data[plot_data['condition_rORp'] == condition]
    label = 'Avoidance' if condition == 0 else 'Approach'
    color = '#E74C3C' if condition == 0 else '#3498DB'
    
    ax.errorbar(cond_data['block_idx'], cond_data['logRT_mean'], 
                yerr=cond_data['logRT_sem'], 
                marker='o', markersize=12, capsize=8, capthick=2.5,
                linewidth=3.5, label=label, color=color, alpha=0.85)

ax.set_xlabel('Block Number', fontsize=28)
ax.set_ylabel('Log(RT) in seconds', fontsize=28)
ax.set_title('Mean Response Time across Blocks by Condition', fontsize=32, pad=20)
ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='best', title='Condition', fontsize=22, title_fontsize=24)
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', labelsize=24)

plt.tight_layout()
plt.savefig(path + 'RESULTS/logRT_by_block_condition.png', dpi=600, bbox_inches='tight')
plt.show()

# ============================================================================
# Figure 2: Bar plot - Mean logRT with condition × block interaction
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

# Prepare data for grouped bar plot
x_pos = np.arange(len(plot_data['block_idx'].unique()))
width = 0.35

risky_data = plot_data[plot_data['condition_rORp'] == 0]
pred_data = plot_data[plot_data['condition_rORp'] == 1]

bars1 = ax.bar(x_pos - width/2, risky_data['logRT_mean'], width, 
               yerr=risky_data['logRT_sem'], capsize=8,
               error_kw={'capthick': 2.5},
               label='Avoidance', color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1.5)

bars2 = ax.bar(x_pos + width/2, pred_data['logRT_mean'], width,
               yerr=pred_data['logRT_sem'], capsize=8,
               error_kw={'capthick': 2.5},
               label='Approach', color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Block Number', fontsize=28)
ax.set_ylabel('Log(RT) in seconds', fontsize=28)
ax.set_title('Response Time by Block and Condition', fontsize=32, pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(risky_data['block_idx'].values.astype(int), fontsize=24)
ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='best', fontsize=22)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=24)

plt.tight_layout()
plt.savefig(path + 'RESULTS/logRT_barplot_by_condition.png', dpi=600, bbox_inches='tight')
plt.show()

# ============================================================================
# Figure 3: Individual subject trajectories
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for idx, condition in enumerate([0, 1]):
    ax = axes[idx]
    cond_label = 'Avoidance' if condition == 0 else 'Approach'
    cond_data = df_logrt[df_logrt['condition_rORp'] == condition]
    
    # Plot individual subject lines with transparency
    for subj in cond_data['Subject_ID'].unique():
        subj_data = cond_data[cond_data['Subject_ID'] == subj]
        subj_means = subj_data.groupby('block_idx')['logRT'].mean()
        ax.plot(subj_means.index, subj_means.values, alpha=0.3, color='gray', linewidth=2)
    
    # Overlay group mean
    group_mean = cond_data.groupby('block_idx')['logRT'].mean()
    ax.plot(group_mean.index, group_mean.values, 'o-', 
            color='#E74C3C' if condition == 0 else '#3498DB', 
            linewidth=4, markersize=12, label='Group Mean', zorder=5)
    
    ax.set_xlabel('Block Number', fontsize=26)
    ax.set_ylabel('Log(RT) in seconds', fontsize=26)
    ax.set_title(f'{cond_label} Condition', fontsize=30, pad=15)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=22)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=22)

plt.suptitle('Individual Subject Trajectories across Blocks', 
             fontsize=32, y=0.995)
plt.tight_layout()
plt.savefig(path + 'RESULTS/logRT_individual_trajectories.png', dpi=600, bbox_inches='tight')
plt.show()

# ============================================================================
# Figure 4: Distribution plots
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

# Prepare data with condition labels for side-by-side violin plot
violin_data_avoid = df_logrt[df_logrt['condition_rORp'] == 0]['logRT'].values
violin_data_approach = df_logrt[df_logrt['condition_rORp'] == 1]['logRT'].values

# Create violin plot using matplotlib
parts = ax.violinplot([violin_data_avoid, violin_data_approach],
                      positions=[0, 1], showmeans=True, showmedians=True, widths=0.6)

# Customize violin plot colors
colors = ['#E74C3C', '#3498DB']
for idx, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[idx])
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')
    pc.set_linewidth(2)

# Customize other components
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
    if partname in parts:
        parts[partname].set_edgecolor('black')
        parts[partname].set_linewidth(2.5)

# Add individual points with jitter for both conditions
for condition, x_pos in enumerate([0, 1]):
    y = df_logrt[df_logrt['condition_rORp'] == condition]['logRT'].values
    x = np.random.normal(x_pos, 0.04, size=len(y))
    ax.scatter(x, y, alpha=0.3, s=60, color='black')

ax.set_ylabel('Log(RT) in seconds', fontsize=28)
ax.set_title('Distribution of Response Times by Condition', fontsize=32, pad=20)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Avoidance', 'Approach'], fontsize=26)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=24)
plt.tight_layout()
plt.savefig(path + 'RESULTS/logRT_distributions.png', dpi=600, bbox_inches='tight')
plt.show()

# ============================================================================
# Figure 5: Interaction plot with block × condition effects
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 9))

# Create pivot table for heatmap
pivot_data = df_logrt.groupby(['block_idx', 'condition_rORp'])['logRT'].mean().unstack()
pivot_data.columns = ['Avoidance', 'Approach']

# Create heatmap
im = ax.imshow(pivot_data.T, aspect='auto', cmap='RdYlBu_r', origin='lower')

# Set ticks and labels
ax.set_xticks(np.arange(len(pivot_data.index)))
ax.set_yticks(np.arange(len(pivot_data.columns)))
ax.set_xticklabels(pivot_data.index.astype(int), fontsize=24)
ax.set_yticklabels(pivot_data.columns, fontsize=24)

# Add values in cells
for i in range(len(pivot_data.columns)):
    for j in range(len(pivot_data.index)):
        text = ax.text(j, i, f'{pivot_data.iloc[j, i]:.3f}',
                      ha="center", va="center", color="black", fontsize=18)

ax.set_xlabel('Block Number', fontsize=28)
ax.set_ylabel('Condition', fontsize=28)
ax.set_title('Mean logRT: Block × Condition Heatmap', fontsize=32, pad=20)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Log(RT) in seconds', fontsize=24)
cbar.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.savefig(path + 'RESULTS/logRT_interaction_heatmap.png', dpi=600, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("VISUALIZATIONS SAVED TO RESULTS/")
print("="*80)
print("✓ logRT_by_block_condition.png - Line plot with error bars")
print("✓ logRT_barplot_by_condition.png - Grouped bar plot")
print("✓ logRT_individual_trajectories.png - Individual subject trajectories")
print("✓ logRT_distributions.png - Violin plots with scatter points")
print("✓ logRT_interaction_heatmap.png - Condition × Block heatmap")

# %% 4. OPTIMAL POLICY VALUES (Q-VALUES) ANALYSIS
# ==============================================================================
# TEST SIGNIFICANCE OF Q-VALUE DIFFERENCES BETWEEN CONDITIONS
# ==============================================================================

print("\n" + "="*80)
print("OPTIMAL POLICY VALUES (ΔQ-VALUES) ANALYSIS: BETWEEN CONDITIONS")
print("="*80)

# Prepare data for Q-value analysis
df_qvals = df[['Subject_ID', 'subj_idx', 'block_idx', 'condition_rORp', 'optimal policy values']].dropna()

# Factorize condition for Q-value analysis
df_qvals['condition_rORp'], condition_labels_q = pd.factorize(df_qvals['condition_rORp'])

# Descriptive statistics by condition
print("\n--- DESCRIPTIVE STATISTICS: OPTIMAL POLICY VALUES BY CONDITION ---")
summary_q = df_qvals.groupby('condition_rORp')['optimal policy values'].agg([
    'count', 'mean', 'std', 'sem'
]).round(4)
print(summary_q)

print("\n--- CONDITION LABELS ---")
for idx, label in enumerate(condition_labels_q):
    print(f"  Condition {idx}: {label}")

# Compute subject-level means for better statistical inference
subject_q_means = df_qvals.groupby(['Subject_ID', 'condition_rORp'])['optimal policy values'].mean().reset_index()
subject_q_means.columns = ['Subject_ID', 'condition_rORp', 'mean_q']

print("\n--- SUBJECT-LEVEL SUMMARY (FIRST 10 ROWS) ---")
print(subject_q_means.head(10))

# Mixed-effects model: optimal policy values ~ condition + (1|Subject_ID)
print("\n--- MIXED-EFFECTS MODEL FOR Q-VALUE DIFFERENCES ---")
print("Formula: optimal_policy_values ~ C(condition_rORp) + (1|Subject_ID)")

model_q = smf.mixedlm(
    "Q('optimal policy values') ~ C(condition_rORp)",
    df_qvals,
    groups=df_qvals["Subject_ID"]
)
result_q = model_q.fit()
print(result_q.summary())

# Extract fixed effects and confidence intervals
print("\n--- FIXED EFFECTS WITH 95% CONFIDENCE INTERVALS ---")
conf_int_q = result_q.conf_int()
conf_int_q.columns = ['Lower CI', 'Upper CI']
params_ci_q = pd.concat([result_q.fe_params.rename('Estimate'), conf_int_q], axis=1)
print(params_ci_q.round(4))

# Extract p-values
print("\n--- P-VALUES FOR FIXED EFFECTS ---")
print(result_q.pvalues.round(4))

# Model fit statistics
print(f"\nModel Fit:")
print(f"  AIC: {result_q.aic:.2f}")
print(f"  BIC: {result_q.bic:.2f}")

# ============================================================================
# Figure 6: Publication-ready plot for optimal policy values (ΔQ-values)
# ============================================================================

print("\n--- GENERATING PUBLICATION-QUALITY VISUALIZATION ---")

# Prepare data for visualization
plot_q_data = df_qvals.groupby('condition_rORp').agg({
    'optimal policy values': ['mean', 'std', 'sem', 'count']
}).reset_index()
plot_q_data.columns = ['condition_rORp', 'q_mean', 'q_std', 'q_sem', 'q_count']
plot_q_data['condition_label'] = plot_q_data['condition_rORp'].map({0: condition_labels_q[0], 1: condition_labels_q[1]})

# Calculate effect size (Cohen's d)
group1 = df_qvals[df_qvals['condition_rORp'] == 0]['optimal policy values']
group2 = df_qvals[df_qvals['condition_rORp'] == 1]['optimal policy values']
pooled_std = np.sqrt(((len(group1)-1)*group1.std()**2 + (len(group2)-1)*group2.std()**2) / (len(group1) + len(group2) - 2))
cohens_d = (group2.mean() - group1.mean()) / pooled_std

print(f"\nEffect Size (Cohen's d): {cohens_d:.4f}")

# Create publication-quality figure with enlarged text
fig, ax = plt.subplots(figsize=(10, 8))

# Set up colors for conditions
colors = ['#E74C3C', '#3498DB']
x_positions = [0, 1]
condition_names = ['Avoidance', 'Approach']

# Plot bar plot with data points
for idx, (x, cond) in enumerate(zip(x_positions, [0, 1])):
    # Get individual subject means
    subj_means = subject_q_means[subject_q_means['condition_rORp'] == cond]['mean_q'].values
    
    # Plot individual points with jitter
    y_jitter = np.random.normal(x, 0.04, size=len(subj_means))
    ax.scatter(y_jitter, subj_means, alpha=0.4, s=120, color=colors[idx], edgecolors='none')
    
    # Plot mean with error bar
    mean_val = plot_q_data[plot_q_data['condition_rORp'] == cond]['q_mean'].values[0]
    sem_val = plot_q_data[plot_q_data['condition_rORp'] == cond]['q_sem'].values[0]
    
    # Plot triangle marker for mean (Delta symbol representation)
    ax.plot(x, mean_val, marker='^', markersize=18, color=colors[idx], 
            markeredgecolor='black', markeredgewidth=2, zorder=5)
    
    # Add error bar
    ax.errorbar(x, mean_val, yerr=sem_val, fmt='none', capsize=10, capthick=3,
                color=colors[idx], elinewidth=3, zorder=4)

# Customize plot with massively enlarged text
ax.set_xticks(x_positions)
ax.set_xticklabels(condition_names, fontsize=26)
ax.set_ylabel(r'$\Delta$$\mathit{Q}$-values', fontsize=28, fontstyle='normal')
ax.set_title(r'$\Delta$$\mathit{Q}$-values' + ' Across Conditions', fontsize=32, pad=20)

# Add grid
ax.grid(True, alpha=0.25, axis='y', linestyle='--')
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Increase font size for ticks
ax.tick_params(axis='y', labelsize=24)
ax.tick_params(axis='x', labelsize=26)

# Add statistical annotation with enlarged text
p_val = result_q.pvalues.iloc[1]  # p-value for condition effect
if p_val < 0.001:
    p_text = "p < 0.001"
elif p_val < 0.01:
    p_text = f"p = {p_val:.2f} **"
elif p_val < 0.05:
    p_text = f"p = {p_val:.2f} *"
else:
    p_text = f"p = {p_val:.2f} ns"

# Add text annotation for statistics with enlarged font
stats_text = f"t = {result_q.tvalues.iloc[1]:.2f}\n{p_text}\nCohen's d = {cohens_d:.2f}"
ax.text(0.5, 0.05, stats_text, transform=ax.transAxes, 
        fontsize=20, verticalalignment='bottom', horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, edgecolor='none', pad=0.8))

plt.tight_layout()
plt.savefig(path + 'RESULTS/delta_q_values_by_condition.png', dpi=600, bbox_inches='tight')
plt.show()

print("\n✓ delta_q_values_by_condition.png - Optimal policy values plot saved")

# %% Check condition difficulty via theoretical MDP value differences (ΔQ)
# ΔQ DIFFICULTY ANALYSIS
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("\n" + "="*80)
print("ΔQ DIFFICULTY ANALYSIS (ITEM-ALIGNED)")
print("="*80)

# =========================================================
# LOAD DATA
# =========================================================

mdp = pd.read_csv("MDP_action_values.csv")
items = pd.read_csv("test_items.csv")

# =========================================================
# DERIVE NUMBER OF FORESTS
# =========================================================

N_FORESTS = len(items)
ROWS_PER_FOREST = 9
DECISION_ROWS = 8

print(f"\nNumber of forests (from items): {N_FORESTS}")

# sanity check
expected_rows = N_FORESTS * ROWS_PER_FOREST
actual_rows = len(mdp)

print(f"Expected MDP rows: {expected_rows}")
print(f"Actual MDP rows:   {actual_rows}")

if actual_rows != expected_rows:
    raise ValueError(
        f"❌ Mismatch: expected {expected_rows} rows in MDP, got {actual_rows}"
    )

# =========================================================
# DETECT MDP COLUMNS
# =========================================================

env1_f = [c for c in mdp.columns if "env1" in c and "fora" in c]
env1_w = [c for c in mdp.columns if "env1" in c and "wait" in c]

env2_f = [c for c in mdp.columns if "env2" in c and "fora" in c]
env2_w = [c for c in mdp.columns if "env2" in c and "wait" in c]

print("\nDetected MDP columns:")
print("env1_f:", env1_f)
print("env1_w:", env1_w)
print("env2_f:", env2_f)
print("env2_w:", env2_w)

# =========================================================
# COMPUTE ΔQ PER FOREST
# =========================================================

forest_abs_means = []
all_abs_dq = []

for f_idx in range(N_FORESTS):

    start = f_idx * ROWS_PER_FOREST
    end   = start + DECISION_ROWS

    forest = mdp.iloc[start:end]

    delta_all = []

    for _, row in forest.iterrows():

        dq_env1 = row[env1_f].values - row[env1_w].values
        dq_env2 = row[env2_f].values - row[env2_w].values

        delta_all.extend(dq_env1)
        delta_all.extend(dq_env2)

    delta_all = np.array(delta_all)
    abs_dq = np.abs(delta_all)

    all_abs_dq.extend(abs_dq)

    forest_abs_means.append({
        "vals_index": f_idx,
        "mean_abs_dq": np.mean(abs_dq)
    })

df_forest = pd.DataFrame(forest_abs_means)

# =========================================================
# GLOBAL HARD THRESHOLD
# =========================================================

all_abs_dq = np.array(all_abs_dq)
threshold = np.percentile(all_abs_dq, 25)

print(f"\nGlobal difficulty threshold: {threshold:.4f}")

# compute hard proportion per forest
hard_props = []

for f_idx in range(N_FORESTS):

    start = f_idx * ROWS_PER_FOREST
    end   = start + DECISION_ROWS

    forest = mdp.iloc[start:end]

    delta_all = []

    for _, row in forest.iterrows():

        dq_env1 = row[env1_f].values - row[env1_w].values
        dq_env2 = row[env2_f].values - row[env2_w].values

        delta_all.extend(dq_env1)
        delta_all.extend(dq_env2)

    abs_dq = np.abs(np.array(delta_all))

    hard_props.append(np.mean(abs_dq < threshold))

df_forest["hard_prop"] = hard_props

# =========================================================
# ALIGN INDEXING
# =========================================================

if items["vals_index"].min() == 1:
    items["vals_index"] -= 1

# =========================================================
# MERGE
# =========================================================

df = items.merge(df_forest, on="vals_index", how="left")

# =========================================================
# CONDITION
# =========================================================

df["condition"] = df["rpDominance"].map({
    "r": "Avoidance",
    "p": "Approach"
})

# =========================================================
# STATISTICS
# =========================================================

avoid = df[df["condition"] == "Avoidance"]["mean_abs_dq"]
approach = df[df["condition"] == "Approach"]["mean_abs_dq"]

u_stat, p_val = stats.mannwhitneyu(avoid, approach, alternative='two-sided')

print("\n--- STATISTICS ---")
print(f"Mann-Whitney U: {u_stat:.2f}, p={p_val:.4f}")

# =========================================================
# PLOT
# =========================================================

colors = {
    "Avoidance": "#E74C3C",
    "Approach": "#3498DB"
}

plt.figure(figsize=(7, 6))

for cond in ["Avoidance", "Approach"]:
    subset = df[df["condition"] == cond]["mean_abs_dq"]
    subset = subset.replace([np.inf, -np.inf], np.nan).dropna()

    plt.hist(
        subset,
        bins=20,
        alpha=0.4,
        density=True,
        label=cond
    )

plt.xlabel(r'Mean $|\Delta Q|$', fontsize=20)
plt.ylabel('Density', fontsize=20)
plt.title('Forest-Level Decision Difficulty (MDP)', fontsize=24)

plt.legend()
plt.grid(alpha=0.2)

plt.tight_layout()
plt.show()

# ECDF Plot
plt.figure(figsize=(9, 6))

for cond, color in [("Avoidance", "#E74C3C"), ("Approach", "#3498DB")]:
    
    # get forest indices for this condition
    forest_ids = df[df["condition"] == cond]["vals_index"].values
    
    all_vals = []

    for f_idx in forest_ids:
        start = f_idx * 9
        end   = start + 8  # 8 decision rows

        forest = mdp.iloc[start:end]

        for _, row in forest.iterrows():
            dq_env1 = row[env1_f].values - row[env1_w].values
            dq_env2 = row[env2_f].values - row[env2_w].values

            all_vals.extend(np.abs(dq_env1))
            all_vals.extend(np.abs(dq_env2))

    # clean + sort
    all_vals = np.array(all_vals)
    all_vals = all_vals[np.isfinite(all_vals)]
    all_vals = np.sort(all_vals)

    y = np.arange(1, len(all_vals)+1) / len(all_vals)

    plt.plot(all_vals, y, label=cond, linewidth=3, color=color)

plt.xlabel(r'$|\Delta Q|$', fontsize=30)  # <- corrected label
plt.ylabel('Cumulative probability', fontsize=30)
plt.title('ECDF: Decision Difficulty Distributions', fontsize=32, pad=15)

plt.legend(fontsize=22)
plt.grid(alpha=0.2)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.tight_layout()
plt.show()

# =========================================================
# VIOLIN PLOT (ROBUST, NO SEABORN)
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

avoid = df[df["condition"] == "Avoidance"]["mean_abs_dq"]
approach = df[df["condition"] == "Approach"]["mean_abs_dq"]

avoid = avoid.replace([np.inf, -np.inf], np.nan).dropna().values
approach = approach.replace([np.inf, -np.inf], np.nan).dropna().values

data = [avoid, approach]
labels = ["Avoidance", "Approach"]
colors = ["#E74C3C", "#3498DB"]

fig, ax = plt.subplots(figsize=(7, 5))

parts = ax.violinplot(
    data,
    positions=[0, 1],
    widths=0.6,
    showmeans=False,
    showmedians=False,
    showextrema=False
)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.5)
    pc.set_edgecolor('black')

# jittered raw data
rng = np.random.default_rng(42)
for i, vals in enumerate(data):
    x = rng.normal(i, 0.05, size=len(vals))
    ax.scatter(x, vals, color="black", alpha=0.5, s=30)

# mean + CI
for i, vals in enumerate(data):
    mean = np.mean(vals)
    sem = np.std(vals, ddof=1) / np.sqrt(len(vals))
    ci = 1.96 * sem

    ax.errorbar(i, mean, yerr=ci, fmt='o', color='black', capsize=6)

ax.set_xticks([0, 1])
ax.set_xticklabels(labels, fontsize=26)
ax.set_ylabel(r'Mean $|\Delta Q|$', fontsize=28)
ax.set_title('Forest-Level Difficulty', fontsize=32, pad=20)
ax.tick_params(axis='y', labelsize=24)

ax.grid(axis='y', alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# =========================================================
# VIOLIN PLOT (STATE-LEVEL |ΔQ|)
# =========================================================

avoid_vals = []
approach_vals = []

for cond, container in [("Avoidance", avoid_vals), ("Approach", approach_vals)]:

    forest_ids = df[df["condition"] == cond]["vals_index"].values

    for f_idx in forest_ids:
        start = f_idx * 9
        end   = start + 8

        forest = mdp.iloc[start:end]

        for _, row in forest.iterrows():
            dq_env1 = row[env1_f].values - row[env1_w].values
            dq_env2 = row[env2_f].values - row[env2_w].values

            container.extend(np.abs(dq_env1))
            container.extend(np.abs(dq_env2))

# convert
avoid_vals = np.array(avoid_vals)
approach_vals = np.array(approach_vals)

# clean
avoid_vals = avoid_vals[np.isfinite(avoid_vals)]
approach_vals = approach_vals[np.isfinite(approach_vals)]

data = [avoid_vals, approach_vals]
labels = ["Avoidance", "Approach"]
colors = ["#E74C3C", "#3498DB"]

fig, ax = plt.subplots(figsize=(8, 6))

parts = ax.violinplot(
    data,
    positions=[0, 1],
    widths=0.6,
    showmeans=False,
    showmedians=False,
    showextrema=False
)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.5)
    pc.set_edgecolor('black')

# mean + CI
for i, vals in enumerate(data):
    mean = np.mean(vals)
    sem = np.std(vals, ddof=1) / np.sqrt(len(vals))
    ci = 1.96 * sem

    ax.errorbar(i, mean, yerr=ci, fmt='o', color='black', capsize=6)

ax.set_xticks([0, 1])
ax.set_xticklabels(labels, fontsize=26)
ax.set_ylabel(r'$|\Delta Q|$', fontsize=28)
ax.set_title('State-Level Decision Difficulty', fontsize=32, pad=20)
ax.tick_params(axis='y', labelsize=24)

ax.grid(axis='y', alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# =========================================================
# SUMMARY
# =========================================================

print("\n--- SUMMARY ---")
print(df.groupby("condition")[["mean_abs_dq", "hard_prop"]].mean())

# =========================================================
# KS TEST (STATE-LEVEL — STRONGER)
# =========================================================
from scipy.stats import ks_2samp

avoid_vals = []
approach_vals = []

for cond, container in [("Avoidance", avoid_vals), ("Approach", approach_vals)]:

    forest_ids = df[df["condition"] == cond]["vals_index"].values

    for f_idx in forest_ids:
        start = f_idx * 9
        end   = start + 8

        forest = mdp.iloc[start:end]

        for _, row in forest.iterrows():
            dq_env1 = row[env1_f].values - row[env1_w].values
            dq_env2 = row[env2_f].values - row[env2_w].values

            container.extend(np.abs(dq_env1))
            container.extend(np.abs(dq_env2))

avoid_vals = np.array(avoid_vals)
approach_vals = np.array(approach_vals)

# clean
avoid_vals = avoid_vals[np.isfinite(avoid_vals)]
approach_vals = approach_vals[np.isfinite(approach_vals)]

ks_stat_s, ks_p_s = ks_2samp(avoid_vals, approach_vals)

print("\n--- KS TEST (STATE LEVEL) ---")
print(f"KS statistic (D): {ks_stat_s:.4f}")
print(f"p-value: {ks_p_s:.4f}")

# %% Detailed RT Analysis 1
""" Test RT differences accross conditions """

import glob

print("\n" + "="*60)
print("Unified Model: MH + OP Features (Standardized, with Condition Effects)")
print("="*60)
print("(All conditions pooled; Continuous variables standardized; OP globally standardized)")
print("(Condition is a main effect + interactions with features)")
print("="*60)

# Define features for the unified condition model
unified_features = [
    '* $\\mathit{p}$ gain',
    '* $\\mathit{r}$ predator',
    '** wait when safe',
    '** binary energy state',
    'optimal policy values'
]

# Load all data (full dataset, no condition filtering - CRITICAL for avoiding separation)
all_data_unified = []
all_responses_unified = []
subject_ids_unified = []
all_conditions_unified = []  # 1 = approach (p), 0 = avoidance (r)
all_block_idx_unified = []  # Block index per observation

s_idx = 0
for itr, fle in enumerate(glob.glob(path + "DATA_clean/test_data.*.CAT.csv")):
    sbj = fle[len(path+"DATA_clean/test_data."):-len(".CAT.csv")]
    dt = pd.read_csv(path + "DATA_clean/test_data." + sbj + ".CAT" + ".csv")
    
    # Add ternary state model
    BNW_state = []
    for index, row in dt.iterrows():
        BNW_state.append(2)
        if row['** binary energy state'] == 1:
            BNW_state[index] = 1
        elif row['** wait when safe'] == 0:
            BNW_state[index] = 3
    dt['ternary state'] = BNW_state


    # ------------------------------------------------------------------------------
    # BLOCK CREATION (PRESERVE TRUE ORDER)
    # ------------------------------------------------------------------------------

    # Previous miniblock value per subject
    prev_mb = dt.groupby("Subject_ID")["miniblock.thisN"].shift()

    # Detect FIRST 0 of each sequence
    dt["block_start"] = (
        (dt["miniblock.thisN"] == 0) & (prev_mb != 0)
    ).astype(int)

    # Count blocks per subject
    dt["block_idx"] = dt.groupby("Subject_ID")["block_start"].cumsum() -1
    
    # Filter but keep BOTH conditions (this is key - no subsetting!)
    dt = dt[dt['foraging T/F NaNs'].isnull() == False]
    dt = dt.reset_index(drop=True)# %% MAIN: Condition Modulation Effect on MH and OP

    # Encode condition: 1 for approach (p), 0 for avoidance (r)
    cond_array = np.array([1 if row["p/r heuristic"] == "['r']" else 0 
                            for _, row in dt.iterrows()])
    
    # Extract block_idx
    block_idx_array = np.array(dt["block_idx"])
    
    all_data_unified.append(dt)
    all_responses_unified.append(np.array(dt["fora_response"]))
    all_conditions_unified.append(cond_array)
    all_block_idx_unified.append(block_idx_array)
    subject_ids_unified.append(np.full(len(dt), s_idx))
    s_idx += 1

n_subjects_unified = len(all_data_unified)

# Prepare data for the unified model
X_list_unified = []
for feat in unified_features:
    feat_data = []
    for subj_data in all_data_unified:
        if feat in subj_data.columns:
            feat_data.append(np.array(subj_data[feat]))
        else:
            feat_data.append(np.full(len(subj_data), np.nan))
    X_list_unified.append(np.concatenate(feat_data))

# Add condition and block_idx as features
cond_unified = np.concatenate(all_conditions_unified)
X_list_unified.append(cond_unified)

block_idx_unified = np.concatenate(all_block_idx_unified)
X_list_unified.append(block_idx_unified)

X_full_unified = np.array(X_list_unified).T
responses_unified = np.concatenate(all_responses_unified)
subj_idx_unified = np.concatenate(subject_ids_unified)

# Create validity mask: no NaNs in features or responses
valid_idx_unified = ~(np.isnan(X_full_unified).any(axis=1) | np.isnan(responses_unified))
X_unified = X_full_unified[valid_idx_unified]
y_unified = responses_unified[valid_idx_unified]
subj_idx_unified = subj_idx_unified[valid_idx_unified]

n_features_unified = len(unified_features)  # 5 main features (no condition as main effect)
n_obs_unified = len(y_unified)

print(f"\nUnified model data (pooled across conditions): {n_obs_unified} observations from {n_subjects_unified} subjects")


# RT Analysis
from scipy.stats import mannwhitneyu

dt['condition'] = cond_array
avoid_rt = dt[dt["condition"] == 1]["logRT"]
approach_rt = dt[dt["condition"] == 0]["logRT"]

avoid_rt = avoid_rt.dropna()
approach_rt = approach_rt.dropna()

u, p = mannwhitneyu(avoid_rt, approach_rt)

print("RT difference:")
print(f"Avoidance mean = {avoid_rt.mean():.3f}")
print(f"Approach mean  = {approach_rt.mean():.3f}")
print(f"U = {u:.2f}, p = {p:.4f}")

# Check if delta Q predicts RT differences between conditions
import statsmodels.formula.api as smf

dt['abs_delta_q'] = abs(dt['optimal policy values'])

model_rt = smf.ols(
    "logRT ~ abs_delta_q * condition",
    data=dt
).fit()

print(model_rt.summary())
