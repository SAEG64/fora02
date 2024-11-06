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
# Get data
data = pd.read_csv(path + "DATA_clean/DATA_group_level/test_data.group_level_datall.csv")
data['resp_count'] = 1
n = len(pd.unique(data['participant']))
# Binning
data['bins'] = pd.qcut(data['optimal policy values'], 7)
data['binNr'] = [(
    float(str(data.bins[:][i]).replace(']','').replace('(','').split(',')[0])+
    float(str(data.bins[:][i]).replace(']','').replace('(','').split(',')[1]))/
    2 for i in range(len(data))]
# Choose experimental condition
# if 1: approach forests
# if 2: avoidance forests
condition = 2
if condition == 1:
    data = data[data["p/r heuristic"] == "['p']"]
elif condition == 2:
    data = data[data["p/r heuristic"] == "['r']"]
# data['binNr'] = [data.iloc[i]['optimal policy values'] if data.iloc[i]['optimal policy values'] == -0.84 or data.iloc[i]['optimal policy values'] == 0.65 else data.iloc[i]['bins'] for i in range(len(data['bins']))]
s_dat = [sbj_file for sbj, sbj_file in data.groupby('participant')] # split subject files

# Compute uncertainty prediction
for i in range(len(s_dat)):
    model = LinearRegression().fit(np.array(s_dat[i]["optimal policy values_uncertain"]).reshape((-1, 1)), np.array(s_dat[i]["logRT"]))
    s_dat[i]['uncertainty_predics'] = model.predict(np.array(s_dat[i]["optimal policy values_uncertain"]).reshape((-1, 1)))
datall = pd.concat(s_dat, axis = 0)

# Aggregate data
top = datall.groupby(['binNr','participant'])[['optimal policy values',
                                            'optimal policy values_uncertain',
                                            'uncertainty_predics',
                                            'logRT']].mean()
top['logRT_std'] =  datall.groupby(['binNr','participant'])['logRT'].sem()
top['resps'] = datall.groupby(['binNr','participant'])['resp_count'].sum()
top_rev = top.groupby('binNr')[['optimal policy values',
                            'optimal policy values_uncertain',
                            'uncertainty_predics',
                            'logRT',
                            'logRT_std',
                            'resps']].mean()

# =============================================================================
# Uncertainty plots
# =============================================================================
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
sns.set_style("white")
sns.set_palette("Paired")

# Canvas
fig, ax = plt.subplots(figsize=(8, 6), 
            dpi = 600)
# fig.tight_layout()
# Plot
plot = sns.regplot(x='optimal policy values', y='logRT', data=top_rev, ci=None, ax=ax,
                    label='responses', scatter_kws={'s':top_rev['resps']*3}, line_kws = {"color": "None"})
ax.errorbar(x='optimal policy values', y='logRT', data=top_rev, 
            yerr = top_rev['logRT_std'], fmt='none', capsize=0, 
            zorder=1, color='C0', label=None)
sns.lineplot(x="optimal policy values", y="uncertainty_predics", 
            data=top_rev, ci=None, ax=ax)
ax.set_title('Policy uncertainty: derivative', loc ='left', size = 32)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.yticks([-0.1, 0, 0.1, 0.2, 0.3])
ax.set(ylim=(-0.2, 0.32))
ax.tick_params(bottom=True, left=True, size=5, direction= "in")
ax.tick_params(axis="x", labelsize=22)
ax.tick_params(axis="y", labelsize=22)
plt.xlabel('optimal policy values bins', fontsize=28)
ax.set_ylabel('Log(RT) in s', fontsize=28, labelpad = -1)

# Canvas
fig, ax = plt.subplots(figsize=(8, 6), 
            dpi = 600)
# fig.tight_layout()
# Plot
plot = sns.regplot(x='optimal policy values_uncertain', y='logRT', data=top_rev, ci=None, ax=ax,
                    label='responses', scatter_kws={'s':top_rev['resps']*3}, line_kws = {"color": "None"})
ax.errorbar(x='optimal policy values_uncertain', y='logRT', data=top_rev, 
            yerr = top_rev['logRT_std'], fmt='none', capsize=0, 
            zorder=1, color='C0', label=None)
sns.lineplot(x="optimal policy values_uncertain", y="uncertainty_predics", 
            data=top_rev, ci=None, ax=ax)
ax.set_title('Linear regression', loc ='left', size = 32)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.yticks([-0.1, 0, 0.1, 0.2, 0.3])
ax.set(ylim=(-0.2, 0.32))
ax.tick_params(bottom=True, left=True, size=5, direction= "in")
ax.tick_params(axis="x", labelsize=22)
ax.tick_params(axis="y", labelsize=22)
plt.xlabel('optimal policy values\nuncertainty values', fontsize=28)
ax.set_ylabel('Log(RT) in s', fontsize=28, labelpad = -1)

# =============================================================================
# Test significances
# =============================================================================
# Load group level data
full_sample = datall
full_sample['condition'] = full_sample['p/r heuristic']
full_sample['uncertainty'] = full_sample['optimal policy values_uncertain']
# MHP uncertainty 
md = smf.mixedlm("logRT ~ uncertainty * condition", full_sample, groups=full_sample["participant"])
mdf = md.fit(method=["lbfgs"])
print(mdf.summary())


