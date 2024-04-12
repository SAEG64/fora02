#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:42:56 2023

@author: sergej
"""

## Select data subset for condition
# 1 for p success dominant heuristic data
# 2 for r threat encounter dominant data
# If anything else: whole dataset is selected
condition = 2

# Requirements
import pandas as pd
import matplotlib.pyplot as plt
import os
import fora_logit_BIC_and_BF

path = os.path.dirname(__file__)+"/"

os.chdir(path)
# from fora_logit_BIC_and_BF import mdlName

## Get data and names
if condition == 1:
    peps = pd.read_csv(path + "/RESULTS/fora_PEPs_looseComp.csv", header=None)
elif condition == 2:
    peps = pd.read_csv(path + "/RESULTS/fora_PEPs_tenseComp.csv", header=None)
else:
    peps = pd.read_csv(path + "/RESULTS/fora_PEPs_datall.csv", header=None)
valu = peps.values.tolist()[0]
name = fora_logit_BIC_and_BF.mdlName

## Plotting
# Figure Size
fig, ax = plt.subplots(figsize =(8, 8))
# Increase x and y labels
ax.tick_params(axis="x", labelsize=34)
ax.tick_params(axis="y", labelsize=34)
ax.tick_params(bottom=True, left=True, size=5, direction= "in")
# Horizontal Bar Plot
ax.barh(name, valu)
# Add Plot Title
ax.set_title('Protected \nexceedance \nprobability (PEP)',
              loc ='left', size = 46)
# ax.set_title('PEP tense \ncompetition',loc ='left', size = 46)
plt.xlabel("", fontsize=40)
# Customize y-ticks
plt.yticks([0, 1,2,3,4,5,6,7,8,9, 10],['','','','','','','','','','multi-heuristic policy',''])
plt.xticks([0, 0.5, 1, 1.05], ["0.0", "0.5", "1.0", ""])
ax.get_yticklabels()[-2].set_color("blue")