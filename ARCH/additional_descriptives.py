#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:55:48 2024

@author: sergej
"""

# =============================================================================
# RT ttest for condition effect
# =============================================================================
import scipy.stats as stats 
import pandas as pd
import os
path = os.path.dirname(__file__)+"/"
os.chdir(path)
# Load and concat data
data = pd.read_csv(path + "DATA_clean/DATA_group_level/test_data.group_level_datall.csv")
data_subs = [data[data['participant'] == pd.unique(data['participant'])[i]] for i in range(len(pd.unique(data['participant'])))]

rt_loose = pd.DataFrame([data[data['condition_rORp'] == 'low threat condition']['logRT'], data[data['condition_rORp'] == 'low threat condition']['participant']]).T
rt_loose_agg = rt_loose.groupby('participant').mean()
rt_tense = pd.DataFrame([data[data['condition_rORp'] == 'high threat condition']['logRT'], data[data['condition_rORp'] == 'high threat condition']['participant']]).T
rt_tense_agg = rt_tense.groupby('participant').mean()

stats.ttest_rel(rt_loose_agg, rt_tense_agg) 

# =============================================================================
# Histogram of number of trials
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
# Distribution of trials/person
x = [len(data_subs[i]) for i in range(len(data_subs))]
counts, bins = np.histogram(x)
plt.stairs(counts, bins)
# print(max([len(data_subs[i]) for i in range(len(data_subs))]))

# =============================================================================
# ttest survival between conditions
# =============================================================================
# Data preprocessing
data_cond = [[data_subs[i][data_subs[i]['condition_rORp'] == 'low threat condition'], data_subs[i][data_subs[i]['condition_rORp'] == 'high threat condition']] for i in range(len(data_subs))]
data_tri_app = [len(data_cond[i][0]) for i in range(len(data_cond))]
data_tri_avo = [len(data_cond[i][1]) for i in range(len(data_cond))]

data_end = [[data_cond[i][0][(data_cond[i][0]['day.thisRepN'] == 7) & (data_cond[i][0]['out_LP'] != 0)], data_cond[i][1][(data_cond[i][1]['day.thisRepN'] == 7) & (data_cond[i][1]['out_LP'] != 0)]] for i in range(len(data_cond))]
# 2 conditions
app_forest_survival = [len(data_end[i][0]) for i in range(len(data_end))] # approach forest success
avo_forest_survival = [len(data_end[i][1]) for i in range(len(data_end))] # avoidance forest success
forest_survival = [app_forest_survival[i] + avo_forest_survival[i] for i in range(len(avo_forest_survival))]
mean_success = np.mean(app_forest_survival + avo_forest_survival)/72
sigm_success = np.std(app_forest_survival + avo_forest_survival)

stats.ttest_rel(app_forest_survival, avo_forest_survival) 

# =============================================================================
# Plot correlation of OP slope with success rate
# =============================================================================
import fora_logit_coefficients
mdlName, mu_b0, mu_b1, si_b0, si_b1, b0_subs, b1_subs, cof_all, min_b0, max_b0, min_b1, max_b1 = fora_logit_coefficients.get_BIC()
op_cap_betas = b1_subs[1]

# Test significance
import statsmodels.api as sm
x = np.array([op_cap_betas]).reshape((-1, 1))
y = np.array(forest_survival)
model = sm.OLS(x,y)
fit = model.fit()

# print(f"intercept: {model.intercept_}")
# print(f"slope: {model.coef_}")
print(fit.summary())

# Plot
import seaborn as sns
fig, ax = plt.subplots(figsize=(8, 6), 
            dpi = 600)
plot = sns.regplot(x=x, y=y, ci=None, ax=ax,
                    label='responses', line_kws = {"color": "red"})
ax.tick_params(bottom=True, left=True, size=5, direction= "in")
ax.tick_params(axis="x", labelsize=22)
ax.tick_params(axis="y", labelsize=22)
plt.xlabel('$\\mathit{OP}$ values + cap \n' + r"$\beta_1$ coefficients (slope)", fontsize=28)
ax.set_ylabel('No. of survived forests', fontsize=28, labelpad = -1)

# # =============================================================================
# # Median split and ttest of success rate
# # =============================================================================
# # Plot success histogram
# counts, bins = np.histogram(y)
# plt.stairs(counts, bins)
# np.median(y)

# # Two performance pools
# betas_low_score = [x[i][0] for i in range(len(x)) if y[i] < np.median(y)]
# betas_high_score = [x[i][0] for i in range(len(x)) if y[i] > np.median(y)]
# # Test significance
# stats.ttest_ind(betas_low_score, betas_high_score)

# # Plot
# betas_low = pd.DataFrame(betas_low_score)
# betas_low['performance'] = 'low'
# betas_high = pd.DataFrame(betas_high_score)
# betas_high['performance'] = 'high'
# dt = pd.concat([betas_low, betas_high])
# dt.columns = ['beta1', 'performance_pool']
# fig, ax = plt.subplots(figsize=(6, 6), dpi = 600)
# ax = sns.boxplot(x='performance_pool', y='beta1', data=dt)
# plt.xlabel('Two performance levels \n(median split)', size = 26)
# plt.ylabel(r"$\beta_1$ coefficients (slope)", size = 26)
# ax.tick_params(axis="x", labelsize=20)
# ax.tick_params(axis="y", labelsize=20, labelbottom = False)
# ax.set_title('Model fits', size = 26)
# ax_labels = [item.get_text() for item in ax.get_xticklabels()]
# ax_labels = [str(round(1-it, 2)) for it in dat_fin["condition"].unique()]
# ax_labels[0] = 'loose forests'
# ax_labels[1] = 'tense forests'
# ax.set_xticklabels(ax_labels)