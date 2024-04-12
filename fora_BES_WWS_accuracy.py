#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:25:04 2024

@author: sergej
"""

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
path = os.path.dirname(__file__)+"/"
os.chdir(path)
# Load and concat data
data = pd.read_csv(path + "DATA_clean/DATA_group_level/test_data.group_level_datall.csv")
data_subs = [data[data['participant'] == pd.unique(data['participant'])[i]] for i in range(len(pd.unique(data['participant'])))]

conds = ['low threat condition', 'high threat condition']
subj_dat = []
for i in range(len(data_subs)):
    data_split = [data_subs[i][data_subs[i]['** binary energy state'] == 1], data_subs[i][data_subs[i]['** wait when safe'] == 0]]
    cond_dat = []
    for j in range(2):
        d_bes = data_split[0][data_split[0]['condition_rORp']== conds[j]]
        d_wws = data_split[1][data_split[1]['condition_rORp']== conds[j]]
        bes, wws = sum(d_bes['foraging T/F'])/len(d_bes), sum(d_wws['foraging T/F'])/len(d_wws)
        cond_dat.append([bes, wws])
    subj_dat.append(cond_dat)

names = ['p fora', 'condition', 'm_type']
bes_loose = [subj_dat[i][0][0] for i in range(29)]
bes_tense = [subj_dat[i][1][0] for i in range(29)]
wws_loose = [subj_dat[i][0][1] for i in range(29)]
wws_tense = [subj_dat[i][1][1] for i in range(29)]
condition = [1 for i in range(29)]+[2 for i in range(29)]+[1 for i in range(29)]+[2 for i in range(29)]
m_type = ['bes' for i in range(29+29)]+['wws' for i in range(29+29)]
dat_fin = pd.DataFrame(list(zip(bes_loose + bes_tense + wws_loose + wws_tense, condition, m_type)))
dat_fin.columns = names

fig, ax = plt.subplots(figsize=(6, 6), dpi = 600)
ax = sns.boxplot(x='condition', y='p fora', data=dat_fin, hue='m_type')
plt.xlabel('Condition', size = 26)
plt.ylabel('$\\mathit{p}$ foraging', size = 26)
ax.tick_params(axis="x", labelsize=20)
ax.tick_params(axis="y", labelsize=20, labelbottom = False)
ax.set_title('Empirical data', size = 26)
ax_labels = [item.get_text() for item in ax.get_xticklabels()]
ax_labels = [str(round(1-it, 2)) for it in dat_fin["condition"].unique()]
ax_labels[0] = 'loose forests'
ax_labels[1] = 'tense forests'
ax.set_xticklabels(ax_labels)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
# Customize legend
handles, labels = ax.get_legend_handles_labels()
labels[0] = "binary e state"
labels[1] = "wait when safe"
l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 0.5), loc=2,
               borderaxespad=0., fontsize=20)
l.set_title('Type of data',prop={'size': 20})
plt.show()


subj_dat = []
for i in range(len(data_subs)):
    data_split = [data_subs[i][data_subs[i]['** binary energy state'] == 1], data_subs[i][data_subs[i]['** wait when safe'] == 0]]
    cond_dat = []
    for j in range(2):
        d_bes = data_split[0][data_split[0]['condition_rORp']== conds[j]]
        d_wws = data_split[1][data_split[1]['condition_rORp']== conds[j]]
        bes, wws = np.mean(d_bes['optimal policy values fit']), np.mean(d_wws['optimal policy values fit'])
        cond_dat.append([bes, wws])
    subj_dat.append(cond_dat)

names = ['p fora', 'condition', 'm_type']
bes_loose = [subj_dat[i][0][0] for i in range(29)]
bes_tense = [subj_dat[i][1][0] for i in range(29)]
wws_loose = [subj_dat[i][0][1] for i in range(29)]
wws_tense = [subj_dat[i][1][1] for i in range(29)]
condition = [1 for i in range(29)]+[2 for i in range(29)]+[1 for i in range(29)]+[2 for i in range(29)]
m_type = ['bes' for i in range(29+29)]+['wws' for i in range(29+29)]
dat_fin = pd.DataFrame(list(zip(bes_loose + bes_tense + wws_loose + wws_tense, condition, m_type)))
dat_fin.columns = names

fig, ax = plt.subplots(figsize=(6, 6), dpi = 600)
ax = sns.boxplot(x='condition', y='p fora', data=dat_fin, hue='m_type')
plt.xlabel('Condition', size = 26)
plt.ylabel('$\\mathit{OP}$ predicted\n$\\mathit{p}$ foraging', size = 26)
ax.tick_params(axis="x", labelsize=20)
ax.tick_params(axis="y", labelsize=20, labelbottom = False)
ax.set_title('Model fits', size = 26)
ax_labels = [item.get_text() for item in ax.get_xticklabels()]
ax_labels = [str(round(1-it, 2)) for it in dat_fin["condition"].unique()]
ax_labels[0] = 'loose forests'
ax_labels[1] = 'tense forests'
ax.set_xticklabels(ax_labels)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
# Customize legend
handles, labels = ax.get_legend_handles_labels()
labels[0] = "binary e state"
labels[1] = "wait when safe"
l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 0.5), loc=2,
               borderaxespad=0., fontsize=20)
l.set_title('Type of data',prop={'size': 20})
plt.show()
