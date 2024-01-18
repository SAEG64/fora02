#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:23:18 2023

@author: sergej
"""
import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib as mpl
import pandas as pd
import os
path = os.path.dirname(__file__)+"/"

# List of model names
mdlName =   ['optimal policy',
              '$\\mathit{p}$ success cor + cap',
              '$\\mathit{p}$ success cor',
              '$\\mathit{p}$ success',
              'risk threat encounter',
              'expected gain based MVT',
              'expected gain naive',
              # 'expected state no bounds',
              # 'expected state with bounds',
              'weather type',
              # 'gain magnitude',
              # 'win stay lose shift',
              # 'continuous energy state',
              'binary energy state']

data = pd.read_csv(path + "DATA_clean/DATA_group_level/test_data.group_level.csv")[mdlName]
matrix = data.corr()
# print(matrix)
fig, ax = plt.subplots(figsize=(10, 8), dpi=600)
cmap = sns.cm.rocket_r
sns.heatmap(matrix, annot=True, yticklabels=True, 
                  xticklabels=True, annot_kws={'size': 12}, cmap = cmap)
ax.tick_params(axis="x", labelsize=24, labelrotation=90)
ax.tick_params(axis="y", labelsize=24, labelrotation=0)
cax = ax.figure.axes[-1]
cax.tick_params(labelsize=24)