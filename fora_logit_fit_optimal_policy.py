#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 09:59:09 2022

@author: sergej
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
path = '/home/sergej/Documents/academics/dnhi/projects/FORA/FORA028t/ANA/CLEAN/'

# Load and concat data
data = pd.read_csv(path + "DATA_clean/DATA_group_level/test_data.group_level.csv")

# Binning
data['bins'] = pd.qcut(data['optimal policy'], 9)

# Aggregate
top = data.groupby(['bins'])['optimal policy', 'fora_response', 
                             'optimal policy fit', 
                             '$\mathit{p}$ success cor fit', 
                             '$\mathit{p}$ success cor + cap fit',
                             'expected gain naive fit'].mean()
# Binning interval averages
top['binNr'] = [(
    float(str(top.index[:][i]).replace(']','').replace('(','').split(',')[0])+
    float(str(top.index[:][i]).replace(']','').replace('(','').split(',')[1]))/
    2 for i in range(0, len(top))]
data['resp_count'] = 1
top['resp_sum'] = data.groupby(['bins'])['resp_count'].sum()
ci = data.groupby(['bins'])['fora_response'].sem(ddof=1)*1.96

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
    x='binNr', y='fora_response', data=top, logistic=True, ci=None,
    ax=ax, label='actual responses', scatter_kws={'s':top['resp_sum']/2}, 
    line_kws = {"color": "None"})
ax.errorbar(
    x='binNr', y='fora_response', data=top, yerr = ci, fmt='none', capsize=0, 
    zorder=1, color='C0', label=None)
for itr, c in enumerate(top.columns[2:-2]):
    nme = 'plot%s' % str(itr+1)
    
    locals()[nme] = sns.lineplot(
        x=top['binNr'], y=top[c], data=top,
        ax=ax, label=c)
    ax.tick_params(bottom=True, left=True, size=5, direction= "in")
# Customize axes
ax.set(ylabel='Foraging likelihood', xlabel='$\mathit{p}$ success corrected')
plt.ylabel("Foraging likelihood", fontsize=30)
plt.xlabel("Optimal policy difference \n forage - wait", fontsize=30)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1],[0, 0.2, 0.4, 0.6, 0.8, 1])
plt.xticks([-0.4, -0.2, 0, 0.2])
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
