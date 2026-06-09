#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 09:59:09 2022

@author: sergej
"""

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
path = os.path.dirname(__file__)+"/"

# Select data subset for condition
# 1 for p success dominant heuristic data (loose competition)
# 2 for r threat encounter dominant data (tense competition)
# If anything else: whole dataset is selected
condition = 0

os.chdir(path)
# Load and preprocess data
data = pd.read_csv(path + "DATA_clean/DATA_group_level/test_data.group_level_datall.csv")
if condition == 1:
    data = pd.read_csv(path + "DATA_clean/DATA_group_level/test_data.group_level_approach.csv")
if condition == 2:
    data = pd.read_csv(path + "DATA_clean/DATA_group_level/test_data.group_level_avoidance.csv")

data['$\mathit{p}$ success'] = data['** $\mathit{p}$ success']
# data['$\mathit{p}$ gain'] = data['* $\mathit{p}$ gain']
# data['binary energy state'] = data['** binary energy state']
# data['wait when safe'] = data['** wait when safe']
data['$\mathit{p}$ success fit'] = data['** $\mathit{p}$ success fit']
# data['$\mathit{p}$ gain fit'] = data['* $\mathit{p}$ gain fit']
# data['binary energy state fit'] = data['** binary energy state fit']
# data['wait when safe fit'] = data['** wait when safe fit']
# data['$\\mathit{r}$ predator fit'] = data['* $\\mathit{r}$ predator fit']
# data['$\\mathit{p}$ gain fit'] = data['* $\\mathit{p}$ gain fit']
data['resp_count'] = 1
n = len(pd.unique(data['participant']))

# Binning
data['bins'] = pd.qcut(data['** $\mathit{p}$ success'], 7)
data['bins'] = [(
    float(str(data.bins[:][i]).replace(']','').replace('(','').split(',')[0])+
    float(str(data.bins[:][i]).replace(']','').replace('(','').split(',')[1]))/
    2 for i in range(0, len(data))]

# Aggregate per bin and per subject
# top = data.groupby(['bins','participant'])[['multi-heuristic policy',
#                                             'fora_response',
#                                             'optimal policy values fit',
#                                             'multi-heuristic policy fit']].mean()
mdls = ['optimal policy values',
         # 'optimal policy values',
        'fora_response',
        # 'optimal policy values fit',
        'optimal policy values fit',
        # 'win stay lose shift fit',
        # 'wait when safe fit',
        # 'binary energy state fit',
        # 'ternary state fit',
        # 'weather type fit',
        # '$\\mathit{r}$ predator fit',
        # '$\\mathit{p}$ gain fit',
        # 'expected gain naive fit',
        # 'marginal value fit',
        '$\mathit{p}$ success fit']
top = data.groupby(['bins','participant'])[mdls].mean()

# Add response counts
top['resp_sum'] = data.groupby(['bins', 'participant'])['resp_count'].sum()
# Aggregate per bin only
top_rev = top.groupby(['bins']).mean()
# Compute standard errors of mean and confidence interval
top_rev['sem'] = np.sqrt((top_rev['fora_response']*(1-top_rev['fora_response']))/top_rev['resp_sum'])
ci = top_rev['sem']*1.96
top_rev = top_rev.reset_index(inplace=False)

## Make plot
# Set publication level params
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
sns.set_style("white")
sns.set_palette("Paired")
# create the figure and axes
fig, ax = plt.subplots(figsize=(6, 6), dpi = 600)
# Create subplots
plot0 = sns.regplot(
    x='bins', y='fora_response', data=top_rev, logistic=True, ci=None, 
    ax=ax, label='actual responses', scatter_kws={'s':top_rev['resp_sum']*3}, 
    line_kws = {"color": "None"})
ax.errorbar(
    x='bins', y='fora_response', data=top_rev, yerr = ci, fmt='none', capsize=0, 
    zorder=1, color='C0', label=None)
for itr, c in enumerate(top_rev.columns[3:-2]):
    nme = 'plot%s' % str(itr+1)
    
    locals()[nme] = sns.lineplot(
        x=top_rev['bins'], y=top_rev[c], data=top_rev, 
        ax=ax, label=c)
    ax.tick_params(bottom=True, left=True, size=5, direction= "in")
# Customize axes
ax.set(ylabel='Foraging likelihood', xlabel='$\mathit{p}$ success binned')
plt.ylabel("Foraging likelihood", fontsize=30)
plt.xlabel("$\mathit{p}$ success binned", fontsize=30)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1],[0.2, 0.4, 0.6, 0.8, 1])
plt.xticks([0.2, 0.4, 0.6])
ax.set(ylim=(0, 1))
ax.tick_params(axis="x", labelsize=24)
ax.tick_params(axis="y", labelsize=24)
# ax.autoscale(enable=True) 
# Legend
# box = ax.get_position()
# ax.set_position([   # Shrink current axis by 20%
#     box.x0, box.y0, 
#     box.width * 0.8, 
#     box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)
# Show plot
plt.show()
