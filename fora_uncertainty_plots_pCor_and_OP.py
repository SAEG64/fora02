#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:24:21 2023

@author: sergej
"""
import glob
import pandas as pd
import matplotlib.pyplot as plt

import os
path = os.path.dirname(__file__)+"/"


# =============================================================================
# Compute uncertainty fit difference
# =============================================================================
subj_data = []
for itr, fle in enumerate(glob.glob(path + "DATA_clean/DATA_fitted/test_data.*.CAT_regress.csv")):
    # Extract participant's data
    sbj = fle[len(path+"DATA_clean/DATA_fitted/test_data."):-len(".CAT_regress.csv")]
    data = pd.read_csv(path + "DATA_clean/DATA_fitted/test_data." + sbj + ".CAT_regress.csv")
    # Binning and aggregate
    data['bins'] = pd.qcut(data['optimal policy'], 8)
    top = data.groupby(['bins'])['optimal policy', '$\mathit{p}$ success cor_uncertain', 'optimal policy_uncertain'].agg(['mean'])
    subj_data.append(top)
dt = pd.concat(subj_data, axis=0).reset_index(drop=True)
dt['bins'] = pd.qcut(dt['optimal policy']['mean'], 8)
dtC = pd.concat([dt['optimal policy'], dt['$\mathit{p}$ success cor_uncertain'], dt['optimal policy_uncertain'], dt['bins']], axis=1).reset_index(drop=True)
mdlName = ["optimal policy", "$\mathit{p}$ success cor_uncertain", "optimal policy uncertain", "bins"]
dtC.columns = mdlName
dtop = dtC.groupby(['bins'])['optimal policy', '$\mathit{p}$ success cor_uncertain', 'optimal policy uncertain'].agg(['mean'])

import matplotlib as mpl
# Set publication level params
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
## Canvas
fig, ax = plt.subplots(figsize=(12, 6), 
            dpi = 600)
plt.plot(dtop['optimal policy'], dtop['$\mathit{p}$ success cor_uncertain'], marker = '.', linewidth=2, markersize=20)
plt.plot(dtop['optimal policy'], dtop['optimal policy uncertain'], marker = '.', linewidth=2.5, markersize=20)
ax.set_title("Uncertainties: derivatives of policies' logit fits",
             loc ='left', size = 32)
labels = ['corrected $\mathit{p}$ success', 'optimal policy']
ax.legend(labels, prop={'size': 24}, bbox_to_anchor=(0.82,0.41))
ax.get_legend().set_title("Policies", prop={'size': 24})
ax.tick_params(axis="x", labelsize=22)
ax.tick_params(axis="y", labelsize=22)
plt.xlabel('MDP value difference forage-wait', fontsize=28)
plt.ylabel('Arbitrary unit', fontsize=28)