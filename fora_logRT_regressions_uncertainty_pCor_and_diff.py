#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:27:49 2023

@author: sergej
"""

import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
import os
path = os.path.dirname(__file__)+"/"


# =============================================================================
# Preprocess data
# =============================================================================
data_raw = []
diff_data = []
for itr, fle in enumerate(glob.glob(path + "DATA_clean/DATA_fitted/test_data.*.CAT_regress.csv")):
    # Extract participant's data
    sbj = fle[len(path+"DATA_clean/DATA_fitted/test_data."):-len(".CAT_regress.csv")]
    data = pd.read_csv(path + "DATA_clean/DATA_fitted/test_data." + sbj + ".CAT_regress.csv")
    # Add subject Nr.
    data['subject_ID'] = itr+1
    # Add 1s to aggregate response counts later
    data["count resp"] = 1
    # Compute_uncertainty difference
    data["uncertainty_diff"] = abs(data["$\mathit{p}$ success cor_uncertain"] - data["optimal policy_uncertain"])
    
    ## Linear regressions
    # linear regression: logRT ~_uncertainty difference
    model = LinearRegression().fit(np.array(data["uncertainty_diff"]).reshape((-1, 1)), np.array(data["logRT"]))
    data["uncertainty_diff_predics"] = model.predict(np.array(data["uncertainty_diff"]).reshape((-1, 1)))
    # linear regression: logRT ~_uncertainty pCor
    model = LinearRegression().fit(np.array(data["$\mathit{p}$ success cor_uncertain"]).reshape((-1, 1)), np.array(data["logRT"]))
    data["$\mathit{p}$_success_cor_uncertain_predict"] = model.predict(np.array(data["$\mathit{p}$ success cor_uncertain"]).reshape((-1, 1)))
    
    # Uncertainty difference binning
    data['bins_diff'] = pd.qcut(data['uncertainty_diff'], 8)
    # Aggregate mean and std
    top = data.groupby(['bins_diff'])['$\mathit{p}$ success cor_uncertain', '$\mathit{p}$_success_cor_uncertain_predict', 'uncertainty_diff', 'uncertainty_diff_predics', 'logRT'].agg(['mean', 'std'])
    # Aggregate response sums
    top['resp_sum'] = data.groupby(['bins_diff'])['count resp'].sum()
    # Append data for uncertainty difference analysis
    diff_data.append(top)    
    data_raw.append(data)
    
## Group level binning uncertainty difference
dt_diff = pd.concat(diff_data, axis=0).reset_index(drop=True)
dt_diff['bins'] = pd.qcut(dt_diff['uncertainty_diff']['mean'], 8)
# Extract varibales of interest
dt_diffC = pd.concat([dt_diff['$\mathit{p}$ success cor_uncertain']['mean'], dt_diff['uncertainty_diff']['mean'], dt_diff['uncertainty_diff_predics']['mean'], dt_diff['$\mathit{p}$_success_cor_uncertain_predict']['mean'], dt_diff['logRT']['mean'], dt_diff['bins'], dt_diff['resp_sum']], axis=1).reset_index(drop=True)
# Flatten data headers
dt_diffC.columns = ['$\mathit{p}$ success cor_uncertain', 'uncertainty_diff', 'uncertainty_diff_predics','$\mathit{p}$_success_cor_uncertain_predict','logRT','bins', 'Nr of responses']
# Aggregate mean, sem and sum
dt_diffCC = dt_diffC.groupby(['bins'])['$\mathit{p}$ success cor_uncertain', '$\mathit{p}$_success_cor_uncertain_predict', 'uncertainty_diff', 'uncertainty_diff_predics','logRT','bins', 'Nr of responses'].agg(['mean', 'sem', 'sum'])
# Concat data
dt_diffFIN = pd.concat([dt_diffCC['$\mathit{p}$ success cor_uncertain']['mean'], dt_diffCC['$\mathit{p}$_success_cor_uncertain_predict']['mean'], dt_diffCC['uncertainty_diff']['mean'], dt_diffCC['uncertainty_diff_predics']['mean'], dt_diffCC['logRT']['mean'], dt_diffCC['logRT']['sem']*1.96, dt_diffCC['Nr of responses']['mean']], axis=1).reset_index(drop=True)
mdlName = ["$\mathit{p}$ foraging success_uncertainty", "$\mathit{p}$ success_uncertainty predicts", "uncertainty_diff", "uncertainty_diff_fits", "mean response time", "response time se", "mean Nr. of responses"]
dt_diffFIN.columns = mdlName

## Group level binning pCor uncertainty
dt_pCor = pd.concat(diff_data, axis=0).reset_index(drop=True)
dt_pCor['bins'] = pd.qcut(dt_pCor['$\mathit{p}$ success cor_uncertain']['mean'], 8)
# Extract variables of interest
dt_pCorC = pd.concat([dt_pCor['$\mathit{p}$ success cor_uncertain']['mean'], dt_pCor['uncertainty_diff']['mean'], dt_pCor['uncertainty_diff_predics']['mean'], dt_pCor['$\mathit{p}$_success_cor_uncertain_predict']['mean'], dt_pCor['logRT']['mean'], dt_pCor['bins'], dt_pCor['resp_sum']], axis=1).reset_index(drop=True)
# Flatten data headers
dt_pCorC.columns = ['$\mathit{p}$ success cor_uncertain', 'uncertainty_diff', 'uncertainty_diff_predics','$\mathit{p}$_success_cor_uncertain_predict','logRT','bins', 'Nr of responses']
# Aggregate mean, sem and sum
dt_pCorCC = dt_pCorC.groupby(['bins'])['$\mathit{p}$ success cor_uncertain', '$\mathit{p}$_success_cor_uncertain_predict', 'uncertainty_diff', 'uncertainty_diff_predics','logRT','bins', 'Nr of responses'].agg(['mean', 'sem', 'sum'])
# Concat data
dt_pCorFIN = pd.concat([dt_pCorCC['$\mathit{p}$ success cor_uncertain']['mean'], dt_pCorCC['$\mathit{p}$_success_cor_uncertain_predict']['mean'], dt_pCorCC['uncertainty_diff']['mean'], dt_pCorCC['uncertainty_diff_predics']['mean'], dt_pCorCC['logRT']['mean'], dt_pCorCC['logRT']['sem']*1.96, dt_pCorCC['Nr of responses']['mean']], axis=1).reset_index(drop=True)
mdlName = ["$\mathit{p}$ foraging success_uncertainty", "$\mathit{p}$ success_uncertainty predicts", "uncertainty_diff", "uncertainty_diff_fits", "mean response time", "response time se", "mean Nr. of responses"]
dt_pCorFIN.columns = mdlName

# =============================================================================
# Plotting linear regressions
# =============================================================================
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

## Uncertainty difference as regressor
# Canvas
fig, ax = plt.subplots(figsize=(12, 6), 
            dpi = 600)
# fig.tight_layout()
# Plot
plot = sns.regplot(x='uncertainty_diff', y='mean response time', data=dt_diffFIN, ci=None, ax=ax,
                   label='responses', scatter_kws={'s':dt_diffFIN['mean Nr. of responses']*5}, line_kws = {"color": "None"})
ax.errorbar(x='uncertainty_diff', y='mean response time', data=dt_diffFIN, 
            yerr = dt_diffFIN['response time se'], fmt='none', capsize=0, 
            zorder=1, color='C0', label=None)
sns.regplot(x="uncertainty_diff", y="mean response time", 
            data=dt_diffFIN, ci=None, ax=ax, label='fit', scatter=False)
# ax.set_title('Linear regression', loc ='left', size = 32)
# ax.legend(loc='center left', prop={'size': 16}, bbox_to_anchor=(0.005,0.15))
plt.yticks([-0.1, 0, 0.1, 0.2, 0.3])
ax.set(ylim=(-0.2, 0.32))
ax.tick_params(bottom=True, left=True, size=5, direction= "in")
ax.tick_params(axis="x", labelsize=22)
ax.tick_params(axis="y", labelsize=22)
plt.xlabel('Absolute uncertainty difference \n $\mathit{p}$ success cor - optimal policy', fontsize=28)
ax.set_ylabel('Log(RT) in s', fontsize=28, labelpad = -1)

## pCor uncertainty as regressor
# Canvas
fig, ax = plt.subplots(figsize=(12, 6), 
            dpi = 600)
# fig.tight_layout()
# Plot
plot = sns.regplot(x='$\mathit{p}$ foraging success_uncertainty', y='mean response time', data=dt_pCorFIN, ci=None, ax=ax,
                   label='responses', scatter_kws={'s':dt_pCorFIN['mean Nr. of responses']*5}, line_kws = {"color": "None"})
ax.errorbar(x='$\mathit{p}$ foraging success_uncertainty', y='mean response time', 
            data=dt_pCorFIN, yerr = dt_pCorFIN['response time se'], fmt='none', 
            capsize=0, zorder=1, color='C0', label=None)
sns.regplot(x="$\mathit{p}$ foraging success_uncertainty", y="mean response time", 
            data=dt_pCorFIN, ci=None, ax=ax, label='fit', scatter=False)
ax.legend(loc='center left', prop={'size': 16}, bbox_to_anchor=(0.04,0.15))
# ax.set_title('Linear regression', loc ='left', size = 32)
# plt.yticks([1, 1.2, 1.4, 1.6])
# ax.set(ylim=(1, 1.6))
ax.tick_params(bottom=True, left=True, size=5, direction= "in")
ax.tick_params(axis="x", labelsize=22)
ax.tick_params(axis="y", labelsize=22)
plt.xlabel('Uncertainty $\mathit{p}$ success cor', fontsize=28)
ax.set_ylabel('Log(RT) in s', fontsize=28, labelpad = 9)

# =============================================================================
# Test significances
# =============================================================================
# Load group level data
full_sample = pd.concat(data_raw, axis=0).reset_index(drop=True)
full_sample['condition'] = full_sample['p/r heuristic']
#_uncertainty difference
md = smf.mixedlm("logRT ~ uncertainty_diff", full_sample, groups=full_sample["subject_ID"])
mdf = md.fit(method=["lbfgs"])
print(mdf.summary())
# p succ cor_uncertainty
full_sample['p_success_cor_uncertain'] = full_sample['$\mathit{p}$ success cor_uncertain']
md = smf.mixedlm("logRT ~ p_success_cor_uncertain", full_sample, groups=full_sample["subject_ID"])
mdf = md.fit(method=["lbfgs"])
print(mdf.summary())