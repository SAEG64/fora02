# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
import os
# import os
path = os.path.dirname(__file__)+"/"
# os.chdir("/home/sergej/Documents/academics/dnhi/projects/FORA/FORA028t/ANA/CLEAN/scripts/")
# from subject_level_BIC_and_lgGrpBF import condition

# Create empty pandas data frame
d = pd.read_csv(path + "DATA_clean/DATA_fitted/test_data.ae11.CAT_regress.csv")
catData = pd.DataFrame(columns = list(d.columns.values))

# Extract subject data and concat stacked frame
for itr, fle in enumerate(glob.glob(path + "DATA_clean/DATA_fitted/test_data.*.CAT_regress.csv")):
    # Get data and add subject ID
    sbj = fle[len(path+"DATA_clean/DATA_fitted/test_data."):-len(".CAT_regress.csv")]
    dt = pd.read_csv(path + "DATA_clean/DATA_fitted/test_data." + sbj + ".CAT_regress.csv")
    dt['subject_ID'] = itr+1
    dt['OP_cap'] = dt['$\mathit{OP}$ values + cap']
    # append catData sheet
    catData = pd.concat([catData, dt], ignore_index = True)

# Rename some variables for compatibility in R
catData['condition_rORp'] = catData['p/r heuristic'].astype('str').map({"['p']": "low threat condition", "['r']": "high threat condition"}).astype(str)
catData['expected_gain'] = catData['expected gain naive']
catData.to_csv(path + 'DATA_clean/DATA_group_level/test_data.group_level.csv')  



# =============================================================================
#   Run some exploratory aanalysis
# =============================================================================
catData_copy = deepcopy(catData)
catData['type'] = 'trade-off'
catData['type'] = ['BES' if catData.iloc[i]['ternary state'] == 1 else catData.iloc[i]['type'] for i in range(len(catData))]
catData['type'] = ['WWS' if catData.iloc[i]['ternary state'] == 3 else catData.iloc[i]['type'] for i in range(len(catData))]

# Filtering
# catData = catData[catData['ternary state'] == 2]
# catData = catData[catData['p/r heuristic'] == "['r']"]
catData['resp_count'] = 1   # Variable to compute sampling rate
catData = catData[['ID_nr','ternary state', 'type', 'optimal policy values','fora_response','logRT','key_resp.rt','resp_count']]
# catData['OP_bin'] = pd.cut(catData['optimal policy values'], bins=10, include_lowest=True)
# Bin ternary state 2 sepparate
cat_det = catData[catData['ternary state'] == 2]
cat_det['OP_bin'] = pd.cut(cat_det['optimal policy values'], bins=7, include_lowest=True)
catData = catData[catData['ternary state'] != 2]
catData['OP_bin'] = pd.cut(catData['optimal policy values'], bins=7, include_lowest=True)
catData = pd.concat([catData, cat_det])
# Filtering
catFilt = catData[np.isnan(catData['fora_response']) == False] # criteria applies by definition

# Aggregated mean data
aggrega = catFilt.groupby(['ID_nr','type','OP_bin']).mean()
aggrega = aggrega.reset_index()
# Count occurances
agg_sum = catFilt.groupby(['ID_nr','type','OP_bin'])['resp_count'].sum()
agg_sum = agg_sum.reset_index()
# Get standard deviations and standard errors
agg_BNW = catFilt.groupby(['ID_nr','type','OP_bin'])['logRT'].std()
agg_BNW = agg_BNW.reset_index()
agg_BNW = agg_BNW.rename(columns = {'logRT': 'logRT_std'})
agg_BNW['logRT_sem'] = agg_BNW['logRT_std']/agg_sum['resp_count']
agg_BNW['RT_sem'] = np.exp(agg_BNW['logRT_sem'])
agg_BNW['choice_sem'] = np.sqrt((aggrega['fora_response']*(1-aggrega['fora_response']))/agg_sum['resp_count'])
agg_BNW['resp_count'] = agg_sum['resp_count']
agg_BNW['choice_mean'] = aggrega['fora_response']
agg_BNW['logRT'] = aggrega['logRT']
agg_BNW['RT'] = aggrega['key_resp.rt']
agg_BNW['optimal policy values'] = aggrega['optimal policy values']
# Final averaging over all subjects
agg_fin = agg_BNW.groupby(['type','OP_bin']).mean()
agg_fin = agg_fin.reset_index()
# Filter nans
agg_fin = agg_fin[np.isnan(agg_fin['optimal policy values']) == False]


# Plot p foraging
fig, ax = plt.subplots(figsize=(6, 6),dpi = 600)
ci = agg_fin['choice_sem']*1.96 # confidence interval
plot_pFora = sns.regplot(
    x='optimal policy values', y='choice_mean', data=agg_fin, logistic=True, ci=None,
    ax=ax, label='', scatter_kws={'s':agg_fin['resp_count']*3}, 
    line_kws = {"color": "None"})
ax.errorbar(
    x='optimal policy values', y='choice_mean', data=agg_fin, yerr = ci, fmt='none', capsize=0, 
    zorder=1, color='C0', label=None)
sns.lineplot(x='optimal policy values', y='choice_mean', data=agg_fin,
        hue='type', ax=ax)
ax.tick_params(bottom=True, left=True, size=5, direction= "in")
plt.ylabel("Foraging likelihood", fontsize=30)
plt.xlabel("Optimal policy value bins", fontsize=30)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
ax.tick_params(axis="x", labelsize=24)
ax.tick_params(axis="y", labelsize=24)

# Plot RT
fig, ax = plt.subplots(figsize=(6, 6),dpi = 600)
ci = agg_fin['logRT_sem']*1.96 # confidence interval
plot_pFora = sns.regplot(
    x='optimal policy values', y='logRT', data=agg_fin, logistic=True, ci=None,
    ax=ax, label='', scatter_kws={'s':agg_fin['resp_count']*3}, 
    line_kws = {"color": "None"})
ax.errorbar(
    x='optimal policy values', y='logRT', data=agg_fin, yerr = ci, fmt='none', capsize=0, 
    zorder=1, color='C0', label=None)
sns.lineplot(x='optimal policy values', y='logRT', data=agg_fin,
        hue='type', ax=ax)
ax.tick_params(bottom=True, left=True, size=5, direction= "in")
plt.ylabel("log(RT)", fontsize=30)
plt.xlabel("Optimal policy value bins", fontsize=30)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
ax.tick_params(axis="x", labelsize=24)
ax.tick_params(axis="y", labelsize=24)

# Trial sampling
catData = deepcopy(catData_copy)
# catData = catData[catData['ternary state'] == 3]
# catData['OP_bin'] = pd.cut(catData['optimal policy values'], bins=14, include_lowest=True)
# catData['binNr'] = [(
#     float(str(catData.OP_bin[:][i]).replace(']','').replace('(','').split(',')[0])+
#     float(str(catData.OP_bin[:][i]).replace(']','').replace('(','').split(',')[1]))/
#     2 for i in range(len(catData))]
fig, ax = plt.subplots(figsize=(6, 6),dpi = 600)
ax.hist(catData['optimal policy values'])#, bins = list(catData['binNr']))
# plt.ylim((0, epis/4))
ax.tick_params(axis='x',which='major',labelsize=24)
ax.tick_params(axis='y',which='major',labelsize=24)
ax.set_xlabel("optimal policy values", fontsize = 30)
ax.set_ylabel("Sampling frequency", fontsize = 30)

fig, ax = plt.subplots(figsize=(6, 6),dpi = 600)
ax.hist(catData['* $\\mathit{r}$ predator'])#, bins = list(catData['binNr']))
# plt.ylim((0, epis/4))
ax.tick_params(axis='x',which='major',labelsize=24)
ax.tick_params(axis='y',which='major',labelsize=24)
ax.set_xlabel("predator risk", fontsize = 30)
ax.set_ylabel("Sampling frequency", fontsize = 30)

fig, ax = plt.subplots(figsize=(6, 6),dpi = 600)
ax.hist(catData['** $\\mathit{p}$ success'])#, bins = list(catData['binNr']))
# plt.ylim((0, epis/4))
ax.tick_params(axis='x',which='major',labelsize=24)
ax.tick_params(axis='y',which='major',labelsize=24)
ax.set_xlabel("success probability", fontsize = 30)
ax.set_ylabel("Sampling frequency", fontsize = 30)

catData = deepcopy(catData_copy)
catData = catData[catData["p/r heuristic"] == "['r']"]
fig, ax = plt.subplots(figsize=(6, 6),dpi = 600)
ax.hist(catData['** $\\mathit{p}$ success'])#, bins = list(catData['binNr']))
# plt.ylim((0, epis/4))
ax.tick_params(axis='x',which='major',labelsize=24)
ax.tick_params(axis='y',which='major',labelsize=24)
ax.set_xlabel("success probability", fontsize = 30)
ax.set_ylabel("Sampling frequency", fontsize = 30)

# Check for correlations within trade-off choices
catData = deepcopy(catData_copy)
catData = catData[catData['ternary state'] == 2]
ops = pd.DataFrame(np.c_[list(catData['OP_value_difference']),list(catData['OP_value_difference_alternative'])])
ops.corr()
ops = pd.DataFrame(np.c_[list(catData['multi-heuristic policy']),list(catData['$\mathit{OP}$ values + cap'])])
ops.corr()
