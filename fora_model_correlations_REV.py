#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:23:18 2023

@author: sergej
"""

# %%
"""This script computes the correlations between the different models """
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as mpl
import pandas as pd
import os
path = os.path.dirname(__file__)+"/"

# List of model names
mdlName =   ['win stay lose shift',
            '** wait when safe',
            '** binary energy state',
            # 'ternary state',
            'weather type',
            '* $\\mathit{r}$ predator',
            '* $\\mathit{p}$ gain',
            'expected gain naive',
            # '** $\\mathit{p}$ success',
            'marginal value',
            # 'multi-heuristic policy',
            # '$\mathit{OP}$ values + cap',
            'optimal policy values']
for i in range(len(mdlName)):
    mdlName[i] = mdlName[i] + ' fit'

data = pd.read_csv(path + "DATA_clean/DATA_group_level/test_data.group_level_univar.csv")[mdlName]
# data['$\mathit{p}$ success'] = data['** $\mathit{p}$ success']
# del data['** $\mathit{p}$ success']
# data['$\mathit{p}$ gain'] = data['* $\mathit{p}$ gain']
# del data['* $\mathit{p}$ gain']
# data['$\mathit{r}$ predator'] = data['* $\\mathit{r}$ predator']
# del data['* $\\mathit{r}$ predator']
# data['binary energy state'] = data['** binary energy state']
# del data['** binary energy state']
# data['wait when safe'] = data['** wait when safe']
# del data['** wait when safe']

Name_REV =   ['win stay lose shift',
            'wait when safe',
            'binary energy state',
            # 'ternary state',
            'weather type',
            '$\\mathit{r}$ predator',
            '$\\mathit{p}$ gain',
            'expected gain',
            # '$\\mathit{p}$ success',
            'marginal value',
            # 'multi-heuristic policy',
            #  '$\mathit{OP}$ values + cap',
            'optimal policy values'][::-1]
data = data.reindex(columns=mdlName[::-1])

# Run and plot correlations
matrix = data.corr()
fig, ax = plt.subplots(figsize=(10, 8), dpi=600)
cmap = sns.cm.rocket_r
sns.heatmap(matrix, annot=True, yticklabels=True, 
                  xticklabels=True, annot_kws={'size': 12}, cmap = cmap)
plt.yticks(np.arange(len(Name_REV))+0.5,Name_REV)
plt.xticks(np.arange(len(Name_REV))+0.5,Name_REV)
ax.tick_params(axis="x", labelsize=24, labelrotation=90)
ax.tick_params(axis="y", labelsize=24, labelrotation=0)
cax = ax.figure.axes[-1]
cax.tick_params(labelsize=24)