#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 09:59:09 2022

@author: sergej
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
path = os.path.dirname(__file__)+"/"
os.chdir(path)
# Load and concat data
data = pd.read_csv(path + "DATA_clean/DATA_group_level/test_data.group_level_datall.csv")
data['$\mathit{p}$ success'] = data['** $\mathit{p}$ success']
data['$\mathit{p}$ gain'] = data['* $\mathit{p}$ gain']
data['binary energy state'] = data['** binary energy state']
data['wait when safe'] = data['** wait when safe']
data['$\mathit{p}$ success fit'] = data['** $\mathit{p}$ success fit']
data['$\mathit{p}$ gain fit'] = data['* $\mathit{p}$ gain fit']
data['binary energy state fit'] = data['** binary energy state fit']
data['wait when safe fit'] = data['** wait when safe fit']
data['resp_count'] = 1
n = len(pd.unique(data['participant']))

# Binning
# data['bins'] = pd.qcut(data['$\mathit{OP}$ values + cap'], 7)
data['bins'] = pd.qcut(data['optimal policy values'], 5)
data['binNr'] = [(
    float(str(data.bins[:][i]).replace(']','').replace('(','').split(',')[0])+
    float(str(data.bins[:][i]).replace(']','').replace('(','').split(',')[1]))/
    2 for i in range(len(data))]
data['binNr'] = [data.iloc[i]['$\mathit{OP}$ values + cap'] if data.iloc[i]['$\mathit{OP}$ values + cap'] == -0.84 or data.iloc[i]['$\mathit{OP}$ values + cap'] == 0.625 else data.iloc[i]['binNr'] for i in range(len(data['binNr']))]

# Aggregate
top = data.groupby(['binNr','participant'])['$\mathit{OP}$ values + cap',
                                            'fora_response',
                                            '$\mathit{OP}$ values + cap fit',
                                            'multi-heuristic policy fit'].mean()
# Add response counts
top['resp_sum'] = data.groupby(['binNr', 'participant'])['resp_count'].sum()
# Aggregate per bin only
top_rev = top.groupby(['binNr']).mean()
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
fig, ax = plt.subplots(figsize=(6, 6),dpi = 600)
# Create subplots
plot0 = sns.regplot(
    x='binNr', y='fora_response', data=top_rev, logistic=True, ci=None,
    ax=ax, label='actual responses', scatter_kws={'s':top_rev['resp_sum']*3}, 
    line_kws = {"color": "None"})
ax.errorbar(
    x='binNr', y='fora_response', data=top_rev, yerr = ci, fmt='none', capsize=0, 
    zorder=1, color='C0', label=None)
for itr, c in enumerate(top_rev.columns[3:-2]):
    nme = 'plot%s' % str(itr+1)
    
    locals()[nme] = sns.lineplot(
        x=top_rev['binNr'], y=top_rev[c], data=top_rev,
        ax=ax, label=c)
    ax.tick_params(bottom=True, left=True, size=5, direction= "in")
# Customize axes
ax.set(ylabel='Foraging likelihood', xlabel='$\\mathit{OP}$ values + cap')
plt.ylabel("Foraging likelihood", fontsize=30)
plt.xlabel("$\\mathit{OP}$ values + cap binned", fontsize=30)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1],[0, 0.2, 0.4, 0.6, 0.8, 1])
plt.xticks([-0.84, -0.4, 0, 0.4])
ax.set(ylim=(0, 1))
ax.tick_params(axis="x", labelsize=24)
ax.tick_params(axis="y", labelsize=24)
# Legend
# box = ax.get_position()
# ax.set_position([   # Shrink current axis by 20%
#     box.x0, box.y0, 
#     box.width * 0.8, 
#     box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)
# Show plot
plt.show()
