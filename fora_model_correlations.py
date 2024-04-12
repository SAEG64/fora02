#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:23:18 2023

@author: sergej
"""
# import seaborn as sns
# import matplotlib.pyplot as plt
# # import matplotlib as mpl
# import pandas as pd
# import os
# path = os.path.dirname(__file__)+"/"

# # List of model names
# mdlName =   ['win stay lose shift',
#             '** wait when safe',
#             '** binary energy state',
#             'weather type',
#             '* $\\mathit{r}$ predator',
#             '* $\\mathit{p}$ gain',
#             'expected gain naive',
#             '** $\\mathit{p}$ success',
#             'marginal value',
#             'multi-heuristic policy',
#             '$\mathit{OP}$ values + cap',
#             'optimal policy values']

# data = pd.read_csv(path + "DATA_clean/DATA_group_level/test_data.group_level.csv")[mdlName]
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

# mdlName =   ['win stay lose shift',
#             'wait when safe',
#             'binary energy state',
#             'weather type',
#             '$\\mathit{r}$ predator',
#             '$\\mathit{p}$ gain',
#             'expected gain naive',
#             '$\\mathit{p}$ success',
#             'marginal value',
#             'multi-heuristic policy',
#              '$\mathit{OP}$ values + cap',
#             'optimal policy values']
# data = data.reindex(columns=mdlName[::-1])

# # Run and plot correlations
# matrix = data.corr()
# fig, ax = plt.subplots(figsize=(10, 8), dpi=600)
# cmap = sns.cm.rocket_r
# sns.heatmap(matrix, annot=True, yticklabels=True, 
#                   xticklabels=True, annot_kws={'size': 12}, cmap = cmap)
# ax.tick_params(axis="x", labelsize=24, labelrotation=90)
# ax.tick_params(axis="y", labelsize=24, labelrotation=0)
# cax = ax.figure.axes[-1]
# cax.tick_params(labelsize=24)

import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib as mpl
import pandas as pd
import os
path = os.path.dirname(__file__)+"/"

# List of model names
mdlName =   ['win stay lose shift',
            '** wait when safe',
            '** binary energy state',
            'weather type',
            '* $\\mathit{r}$ predator',
            '* $\\mathit{p}$ gain',
            'expected gain naive',
            '** $\\mathit{p}$ success',
            'marginal value',
            'multi-heuristic policy',
            '$\mathit{OP}$ values + cap',
            'optimal policy values']

data = pd.read_csv(path + "DATA_clean/DATA_group_level/test_data.group_level_datall.csv")[mdlName]
data['$\mathit{p}$ success'] = data['** $\mathit{p}$ success']
del data['** $\mathit{p}$ success']
data['$\mathit{p}$ gain'] = data['* $\mathit{p}$ gain']
del data['* $\mathit{p}$ gain']
data['$\mathit{r}$ predator'] = data['* $\\mathit{r}$ predator']
del data['* $\\mathit{r}$ predator']
data['binary energy state'] = data['** binary energy state']
del data['** binary energy state']
data['wait when safe'] = data['** wait when safe']
del data['** wait when safe']

mdlName =   ['win stay lose shift',
            'wait when safe',
            'binary energy state',
            'weather type',
            '$\\mathit{r}$ predator',
            '$\\mathit{p}$ gain',
            'expected gain naive',
            '$\\mathit{p}$ success',
            'marginal value',
            'multi-heuristic policy',
             '$\mathit{OP}$ values + cap',
            'optimal policy values']
data = data.reindex(columns=mdlName[::-1])

# Run and plot correlations
matrix = data.corr()
fig, ax = plt.subplots(figsize=(10, 8), dpi=600)
cmap = sns.cm.rocket_r
sns.heatmap(matrix, annot=True, yticklabels=True, 
                  xticklabels=True, annot_kws={'size': 12}, cmap = cmap)
ax.tick_params(axis="x", labelsize=24, labelrotation=90)
ax.tick_params(axis="y", labelsize=24, labelrotation=0)
cax = ax.figure.axes[-1]
cax.tick_params(labelsize=24)