#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:22:08 2023

@author: sergej
"""
# %%
# Requirements
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from scipy import stats
path = os.path.dirname(__file__)+"/"


coefs_cond1_op = [[],[]]
coefs_cond2_op = [[],[]]
coefs_cond1_mhp = [[],[]]
coefs_cond2_mhp = [[],[]]

for itr, fle in enumerate(glob.glob(path + "DATA_clean/DATA_fitted/test_data.*.CAT_regress.csv")):
    # Extract participant's data
    sbj = fle[len(path+"DATA_clean/DATA_fitted/test_data."):-len(".CAT_regress.csv")]
    data = pd.read_csv(path + "DATA_clean/DATA_fitted/test_data." + sbj + ".CAT_regress.csv")
    # Separate Conditions
    data_cond1 = data[data['p/r heuristic'] == "['p']"]
    data_cond2 = data[data['p/r heuristic'] == "['r']"]
    
    ## Get coefs for OP_cap in Condition 1
    # Get model
    model = data_cond1['$\mathit{OP}$ values + cap']
    # Get responses
    respo = np.array(data_cond1["fora_response"])
    # Add constant for intercept
    model = np.array(model)
    model = sm.add_constant(model)
    # Run logit
    mdl = sm.Logit(respo, model)
    result = mdl.fit()
    # Append model coefs
    coefs_cond1_op[0].append(result.params[0])
    coefs_cond1_op[1].append(result.params[1])
    
    ## Get coefs for OP_cap in Condition 2
    # Get model
    model = data_cond2['$\mathit{OP}$ values + cap']
    # Get responses
    respo = np.array(data_cond2["fora_response"])
    
    model = np.array(model)
    model = sm.add_constant(model)
    # Run logit
    mdl = sm.Logit(respo, model)
    result = mdl.fit()
    # Append model coefs
    coefs_cond2_op[0].append(result.params[0])
    coefs_cond2_op[1].append(result.params[1])
    
    ## Get coefs for MHP in Condition 1
    # Get model
    model = data_cond1['multi-heuristic policy']
    # Get responses
    respo = np.array(data_cond1["fora_response"])
    # Add constant for intercept
    model = np.array(model)
    model = sm.add_constant(model)
    # Run logit
    mdl = sm.Logit(respo, model)
    result = mdl.fit()
    # Append model coefs
    coefs_cond1_mhp[0].append(result.params[0])
    coefs_cond1_mhp[1].append(result.params[1])
    
    ## Get coefs for MHP in Condition 2
    # Get model
    model = data_cond2['multi-heuristic policy']
    # Get responses
    respo = np.array(data_cond2["fora_response"])
    
    model = np.array(model)
    model = sm.add_constant(model)
    # Run logit
    mdl = sm.Logit(respo, model)
    result = mdl.fit()
    # Append model coefs
    coefs_cond2_mhp[0].append(result.params[0])
    coefs_cond2_mhp[1].append(result.params[1])

# Prepare data for plotting - OP_cap coefficients
df11_op = pd.DataFrame(coefs_cond1_op[0])
df11_op['Coefficients'] = r"$\beta_0$"
df11_op['Model'] = "OP_cap"
df12_op = pd.DataFrame(coefs_cond1_op[1])
df12_op['Coefficients'] = r"$\beta_1$"
df12_op['Model'] = "OP_cap"
df3_op = pd.concat([df11_op, df12_op], axis = 0)
df3_op['Conditions'] = "Low Threat"

df21_op = pd.DataFrame(coefs_cond2_op[0])
df21_op['Coefficients'] = r"$\beta_0$"
df21_op['Model'] = "OP_cap"
df22_op = pd.DataFrame(coefs_cond2_op[1])
df22_op['Coefficients'] = r"$\beta_1$"
df22_op['Model'] = "OP_cap"
df4_op = pd.concat([df21_op, df22_op], axis = 0)
df4_op['Conditions'] = "High Threat"

# Prepare data for plotting - MHP coefficients
df11_mhp = pd.DataFrame(coefs_cond1_mhp[0])
df11_mhp['Coefficients'] = r"$\beta_0$"
df11_mhp['Model'] = "MHP"
df12_mhp = pd.DataFrame(coefs_cond1_mhp[1])
df12_mhp['Coefficients'] = r"$\beta_1$"
df12_mhp['Model'] = "MHP"
df3_mhp = pd.concat([df11_mhp, df12_mhp], axis = 0)
df3_mhp['Conditions'] = "Low Threat"

df21_mhp = pd.DataFrame(coefs_cond2_mhp[0])
df21_mhp['Coefficients'] = r"$\beta_0$"
df21_mhp['Model'] = "MHP"
df22_mhp = pd.DataFrame(coefs_cond2_mhp[1])
df22_mhp['Coefficients'] = r"$\beta_1$"
df22_mhp['Model'] = "MHP"
df4_mhp = pd.concat([df21_mhp, df22_mhp], axis = 0)
df4_mhp['Conditions'] = "High Threat"

# Combine all data
df_stack = pd.concat([df3_op, df4_op, df3_mhp, df4_mhp], axis = 0).reset_index(drop=True)
df_stack = df_stack.rename(columns = {0:'β distributions'})

# Filter to only include beta1 (slope) coefficients
df_beta1_only = df_stack[df_stack['Coefficients'] == r"$\beta_1$"]

## Plotting
# Set publication level params for plotting
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
sns.set_style("white")
sns.set_palette("Paired")

# Calculate mean values for each group
mean_data = df_beta1_only.groupby(['Conditions', 'Model'])['β distributions'].mean().reset_index()

# Create side-by-side box plots for both models
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=600)

# OP_cap plot
df_op_beta1 = df_beta1_only[df_beta1_only['Model'] == 'OP_cap']
sns.boxplot(x="Conditions", y="β distributions",
            data=df_op_beta1, showcaps=False, boxprops={'alpha': 0.4}, 
            color="blue", ax=ax1)

# Add stripplot for OP_cap
sns.stripplot(x="Conditions", y="β distributions",
            data=df_op_beta1, color="blue", ax=ax1)

# Calculate mean values for OP_cap beta1
mean_data_op = df_op_beta1.groupby(['Conditions'])['β distributions'].mean().reset_index()
x_positions = [0, 1]  # Low Threat and High Threat
y_values_op = mean_data_op['β distributions'].values

# Add mean line for OP_cap
ax1.plot(x_positions, y_values_op, color="blue", marker='o', markersize=8, 
        linewidth=3, linestyle='-', zorder=10)

ax1.set_title('OP_cap β1 Interaction Effect',
             loc='left', size=20)
ax1.tick_params(bottom=True, left=True, size=5, direction="in")
ax1.tick_params(axis="x", labelsize=14)
ax1.tick_params(axis="y", labelsize=14)
ax1.set_ylabel("β1 distributions", fontsize=16)
ax1.set_xlabel("Threat level", fontsize=16)

# MHP plot
df_mhp_beta1 = df_beta1_only[df_beta1_only['Model'] == 'MHP']
sns.boxplot(x="Conditions", y="β distributions",
            data=df_mhp_beta1, showcaps=False, boxprops={'alpha': 0.4}, 
            color="green", ax=ax2)

# Add stripplot for MHP
sns.stripplot(x="Conditions", y="β distributions",
            data=df_mhp_beta1, color="green", ax=ax2)

# Calculate mean values for MHP beta1
mean_data_mhp = df_mhp_beta1.groupby(['Conditions'])['β distributions'].mean().reset_index()
y_values_mhp = mean_data_mhp['β distributions'].values

# Add mean line for MHP
ax2.plot(x_positions, y_values_mhp, color="green", marker='o', markersize=8, 
        linewidth=3, linestyle='-', zorder=10)

ax2.set_title('MHP β1 Interaction Effect',
             loc='left', size=20)
ax2.tick_params(bottom=True, left=True, size=5, direction="in")
ax2.tick_params(axis="x", labelsize=14)
ax2.tick_params(axis="y", labelsize=14)
ax2.set_ylabel("β1 distributions", fontsize=16)
ax2.set_xlabel("Threat level", fontsize=16)

plt.tight_layout()
plt.show()

# Print summary statistics
print("="*60)
print("SUMMARY STATISTICS FOR β1 COEFFICIENTS")
print("="*60)

print("\nOP_cap Model:")
print(f"Low Threat - Mean: {mean_data_op.iloc[0]['β distributions']:.4f}")
print(f"High Threat - Mean: {mean_data_op.iloc[1]['β distributions']:.4f}")
print(f"Difference (High - Low): {mean_data_op.iloc[1]['β distributions'] - mean_data_op.iloc[0]['β distributions']:.4f}")

print("\nMHP Model:")
print(f"Low Threat - Mean: {mean_data_mhp.iloc[0]['β distributions']:.4f}")
print(f"High Threat - Mean: {mean_data_mhp.iloc[1]['β distributions']:.4f}")
print(f"Difference (High - Low): {mean_data_mhp.iloc[1]['β distributions'] - mean_data_mhp.iloc[0]['β distributions']:.4f}")

# Perform paired t-tests for both models
print("\n" + "="*60)
print("STATISTICAL TESTS")
print("="*60)

# Perform GLMM to test interaction effect
# First, reshape data to long format for GLMM analysis
import statsmodels.formula.api as smf
from scipy import stats

# Create a long-format dataframe with all individual trial data
all_data = []
for itr, fle in enumerate(glob.glob(path + "DATA_clean/DATA_fitted/test_data.*.CAT_regress.csv")):
    sbj = fle[len(path+"DATA_clean/DATA_fitted/test_data."):-len(".CAT_regress.csv")]
    data = pd.read_csv(path + "DATA_clean/DATA_fitted/test_data." + sbj + ".CAT_regress.csv")
    data['subject'] = sbj
    all_data.append(data)

# Combine all subject data
combined_data = pd.concat(all_data, ignore_index=True)

# Create numeric condition variable (0 = Low Threat, 1 = High Threat)
combined_data['condition_num'] = (combined_data['p/r heuristic'] == "['r']").astype(int)

# Perform paired t-tests for beta1 coefficients between conditions
# OP_cap model
beta1_op_low = df_op_beta1[df_op_beta1['Conditions'] == 'Low Threat']['β distributions'].values
beta1_op_high = df_op_beta1[df_op_beta1['Conditions'] == 'High Threat']['β distributions'].values

# MHP model
beta1_mhp_low = df_mhp_beta1[df_mhp_beta1['Conditions'] == 'Low Threat']['β distributions'].values
beta1_mhp_high = df_mhp_beta1[df_mhp_beta1['Conditions'] == 'High Threat']['β distributions'].values

# Perform paired t-tests
t_stat_op, p_value_op = stats.ttest_rel(beta1_op_low, beta1_op_high)
t_stat_mhp, p_value_mhp = stats.ttest_rel(beta1_mhp_low, beta1_mhp_high)

print(f"\nPaired t-test for OP_cap β1 coefficients:")
print(f"t-statistic: {t_stat_op:.4f}")
print(f"p-value: {p_value_op:.4f}")
print(f"Mean β1 Low Threat: {np.mean(beta1_op_low):.4f}")
print(f"Mean β1 High Threat: {np.mean(beta1_op_high):.4f}")

print(f"\nPaired t-test for MHP β1 coefficients:")
print(f"t-statistic: {t_stat_mhp:.4f}")
print(f"p-value: {p_value_mhp:.4f}")
print(f"Mean β1 Low Threat: {np.mean(beta1_mhp_low):.4f}")
print(f"Mean β1 High Threat: {np.mean(beta1_mhp_high):.4f}")

# Fit GLMM with interaction term for OP_cap
print("\n" + "="*60)
print("GLMM ANALYSIS FOR OP_cap MODEL")
print("="*60)

glmm_full_op = smf.mixedlm("fora_response ~ C(condition_num) * Q('$\mathit{OP}$ values + cap')", 
                          combined_data, 
                          groups=combined_data["subject"],
                          re_formula="~ Q('$\mathit{OP}$ values + cap')")
glmm_result_full_op = glmm_full_op.fit()

glmm_reduced_op = smf.mixedlm("fora_response ~ C(condition_num) + Q('$\mathit{OP}$ values + cap')", 
                             combined_data, 
                             groups=combined_data["subject"],
                             re_formula="~ Q('$\mathit{OP}$ values + cap')")
glmm_result_reduced_op = glmm_reduced_op.fit()

# Calculate Type III test for OP_cap
likelihood_ratio_op = -2 * (glmm_result_reduced_op.llf - glmm_result_full_op.llf)
p_value_type3_op = stats.chi2.sf(likelihood_ratio_op, df=1)

print(f"OP_cap Interaction - Type III Test:")
print(f"Likelihood Ratio: {likelihood_ratio_op:.4f}")
print(f"p-value: {p_value_type3_op:.4f}")

# Fit GLMM with interaction term for MHP
print("\n" + "="*60)
print("GLMM ANALYSIS FOR MHP MODEL")
print("="*60)

glmm_full_mhp = smf.mixedlm("fora_response ~ C(condition_num) * Q('multi-heuristic policy')", 
                           combined_data, 
                           groups=combined_data["subject"],
                           re_formula="~ Q('multi-heuristic policy')")
glmm_result_full_mhp = glmm_full_mhp.fit()

glmm_reduced_mhp = smf.mixedlm("fora_response ~ C(condition_num) + Q('multi-heuristic policy')", 
                              combined_data, 
                              groups=combined_data["subject"],
                              re_formula="~ Q('multi-heuristic policy')")
glmm_result_reduced_mhp = glmm_reduced_mhp.fit()

# Calculate Type III test for MHP
likelihood_ratio_mhp = -2 * (glmm_result_reduced_mhp.llf - glmm_result_full_mhp.llf)
p_value_type3_mhp = stats.chi2.sf(likelihood_ratio_mhp, df=1)

print(f"MHP Interaction - Type III Test:")
print(f"Likelihood Ratio: {likelihood_ratio_mhp:.4f}")
print(f"p-value: {p_value_type3_mhp:.4f}")

# Summary comparison
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print(f"OP_cap interaction p-value: {p_value_type3_op:.4f}")
print(f"MHP interaction p-value: {p_value_type3_mhp:.4f}")

if p_value_type3_op < 0.05:
    print("OP_cap: Significant interaction effect detected")
else:
    print("OP_cap: No significant interaction effect")
    
if p_value_type3_mhp < 0.05:
    print("MHP: Significant interaction effect detected")
else:
    print("MHP: No significant interaction effect")