#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:37:28 2023

@author: sergej
"""

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import numpy as np
import pandas as pd
import math
from copy import deepcopy
import statsmodels.api as sm
import scipy.stats as stats
import statistics
import os
path = os.path.dirname(__file__)+"/"


## Set publication level params for plotting
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
sns.set_style("white")
sns.set_palette("Paired")

## Select data subset
#  whole data set or condition
#  according to model comparison script
from fora_logit_BIC_and_BF import condition

## Requirements for simulation
frst = pd.read_csv(path + "test_items.csv")
mdpV = pd.read_csv(path + "MDP_action_value_difference.csv")
frst['expected_gain_left'] = frst['pLeft_correct']*frst["value_succLeft"]+frst["r_threatL"]*(-3)+(1-frst['pLeft_correct'])*(-2)
frst['expected_gain_right'] = frst['pRight_correct']*frst["value_succRight"]+frst["r_threatR"]*(-3)+(1-frst['pRight_correct'])*(-2)
os.chdir(path)
import fora_logit_coefficients
mdlName, mu_b0, mu_b1, si_b0, si_b1, b0_subs, b1_subs, cof_all, min_b0, max_b0, min_b1, max_b1 = fora_logit_coefficients.get_BIC()

# Filter conditions
if condition == 1:
    frst = frst[frst["rpDominance"] == 'p']
if condition == 2:
    frst = frst[frst["rpDominance"] == 'r']
    
# Initial life points
def ini_lp():
    # lp = random.choice([3,5])
    lp = random.choice((4,5))
    return lp
# Rewards
def reward(lp_cur):
    if lp_cur > 0:
        rew = 1
    else:
        rew = 0
    return rew
# activation function
def activate(dv):
    p_fora = 1/(1+math.e**(-dv))
    return p_fora

# Parameters
epis = 1000
nDay = 8
nSta = 7

## Run simulations
confusion_matrix = []
survival_distrib = dict()
for itr, mod in enumerate(mdlName):
    # Initiate list for survival distribution
    rew_distr = []
    # Initiate lists for parameter recovery
    choice_b0 = []
    choice_b1 = []
    modell_b0 = []
    modell_b1 = []
    # Get model name
    model_name = mod
    # Classification per episode
    winr = np.zeros((len(mdlName)))
    for epi in range(epis):
        
        # Draw logistic betas from empirical distribution (truncated)
        lower_b0 = min_b0[itr]
        upper_b0 = max_b0[itr]
        lower_b1 = min_b1[itr]
        upper_b1 = max_b1[itr]
        b0 = stats.truncnorm.rvs((lower_b0 - mu_b0[itr]) / si_b0[itr], (upper_b0 - mu_b0[itr]) / si_b0[itr], loc=mu_b0[itr], scale=si_b0[itr])
        b1 = stats.truncnorm.rvs((lower_b1 - mu_b1[itr]) / si_b1[itr], (upper_b1 - mu_b1[itr]) / si_b1[itr], loc=mu_b1[itr], scale=si_b1[itr])
        # b0 = np.random.normal(mu_b0[itr], si_b0[itr], size = 1)[0]
        # b1 = np.random.normal(mu_b1[itr], si_b1[itr], size = 1)[0]  
        # Append choice-betas for parameter recovery
        choice_b0.append(b0)
        choice_b1.append(b1)
        # Initialize predictions, responses and starvation counts
        mods = [[] for i in range(0, len(mdlName))]
        resp = []
        died = []
        for frs in range(len(frst)):
            # Initialize lp
            lp_cur = ini_lp()
            # Dummy for LP past
            lp_past = np.nan
            # Time-point loop
            for day in range(nDay): 
                # break frs if dead
                if lp_cur == 0:
                    died.append(True)
                    break
                else:
                    # Get weather type
                    env = random.choice((0,1))
                    
                    ## Model variables
                    if env == 0:
                        expected_gain = frst.iloc[frs]["expected_gain_left"]
                        p_uncorrected = frst.iloc[frs]["pLeft_uncorrected"]
                        p_corrected = frst.iloc[frs]["pLeft_correct"]
                        r = frst.iloc[frs]["r_threatL"]
                    else:
                        expected_gain = frst.iloc[frs]["expected_gain_right"]
                        p_uncorrected = frst.iloc[frs]["pRight_uncorrected"]
                        p_corrected = frst.iloc[frs]["pRight_correct"]
                        r = frst.iloc[frs]["r_threatL"]
                    p_corrected_capped = deepcopy(p_corrected)
                    optimal_policy = mdpV.iloc[8-day+9*frs,int(lp_cur+nSta*env)]
                    optimal_policy_cap = mdpV.iloc[8-day+9*frs,int(lp_cur+nSta*env)]
                    if lp_cur == 1:
                        p_corrected_capped = 1
                        optimal_policy_cap = 0.625
                    elif lp_cur > nDay+1-day:
                        p_corrected_capped = 0
                        optimal_policy_cap = -0.84
                    # Model values
                    mods[0].append(optimal_policy)
                    mods[1].append(optimal_policy_cap)
                    mods[2].append(p_corrected_capped)
                    mods[3].append(p_corrected)
                    mods[4].append(p_uncorrected)
                    mods[5].append(expected_gain)
                    ## 'Actual' (resp. simulated) choices
                    if mod == 'optimal policy values':
                        x = optimal_policy
                    if mod == '$\\mathit{OP}$ values + cap':
                        x = optimal_policy_cap
                    elif mod == 'multi-heuristic policy':
                        x = p_corrected_capped
                    elif mod ==  '** $\\mathit{p}$ success':
                        x = p_corrected
                    elif mod == '* $\\mathit{p}$ gain':
                        x = p_uncorrected
                    elif mod == 'expected gain naive':
                        x = expected_gain
                    if np.isnan(x) == False:
                        # Decision variable
                        dv = b0 + b1*x
                        # Run activation function
                        p_fora = activate(dv)
                        # Make choice
                        if random.random() <= p_fora:
                            action = 1
                        else:
                            action = 0
                    else:
                        # If choice = Nane (e.g. first WSLS iteration)
                        dv = np.nan
                        p_fora = np.nan
                        action = np.random.choice([1, 0])
                    resp.append(action)
                    
                    ## Choice outcome
                    lp_past = lp_cur
                    if action == 1:
                        succ = random.random() <= p_uncorrected
                        suff = random.random() <= r
                        wait = 0
                        if suff == True:
                            lp_cur -= 3
                        else:
                            if succ == True:
                                lp_cur += frst.iloc[frs]["value_succLeft"]
                            else:
                                lp_cur -= 2
                    else:
                        succ = 0
                        suff = 0
                        wait = 1
                        lp_cur -= wait
                    lp_cur = np.clip(np.array(lp_cur), a_min = 0, a_max = 6)
        # Append survival rate per episode (transformed from death count)
        rew_distr.append(len(frst)-sum(died))
                    
        # Model inference
        BIC = []
        for jtr, mad in enumerate(mdlName):
            y_true = [np.nan if np.isnan(mods[jtr][i]) else resp[i] for i in range(0, len(resp))]
            y_true = [ele for ele in resp if np.isnan(ele) == False]
            y_true = np.array(resp)
            x_true = np.array([ele for ele in mods[jtr] if np.isnan(ele) == False])
            x_true = sm.add_constant(np.array(mods[jtr]))
            model = sm.Logit(y_true, x_true)
            fitti = model.fit()
            BIC.append(fitti.bic)
            if mod == mad:
                recovery_model = mod
                modell_b0.append(fitti.params[0])
                modell_b1.append(fitti.params[1])
        # Count each time a model wins
        winr[np.argmin(BIC)] += 1
        
    ## Plot simulated reward distributions
    fig, ax = plt.subplots(figsize=(10, 8), dpi = 600)
    ax.hist(rew_distr, bins = list(range(len(frst))))
    plt.ylim((0, epis/4))
    ax.tick_params(axis='x',which='major',labelsize=24)
    ax.tick_params(axis='y',which='major',labelsize=24)
    ax.set_title(mod, fontsize = 38)
    ax.set_xlabel("Points achieved", fontsize = 30)
    ax.set_ylabel("Frequency", fontsize = 30)
    survival_distrib[mod] = {"reward mean": np.mean(rew_distr), "reward std": statistics.stdev(rew_distr)}
        
    ## Create confusion matrix
    winr_proportiona = [round(winr[_]/epis, 3) for _  in range(len(winr))]
    confusion_matrix.append(winr/epis)
    
    ## Parameter recovery
    beta0 = pd.DataFrame(list(zip(modell_b0, choice_b0)), columns =  ["Beta0 model", "Beta0 target"])
    beta1 = pd.DataFrame(list(zip(modell_b1, choice_b1)), columns =  ["Beta0 model", "Beta1 target"])
    # Correlation beta0
    stats.pearsonr(modell_b0, choice_b0)
    sns_plot = sns.lmplot(data = beta0, x="Beta0 model", y="Beta0 target")
    plt.gcf().set_size_inches(6, 6)
    plt.tick_params(axis="x", labelsize=24)
    plt.tick_params(axis="y", labelsize=24)
    plt.title(mod, fontsize = 38)
    plt.ylabel(r"$\beta_0$ target", fontsize=30)
    plt.xlabel(r"$\beta_0$ model", fontsize=30)
    # sns_plot.figure.savefig('/home/sergej/Documents/academics/dnhi/projects/AAA/FORA02/RESULTS/figures_intern/parameter_recovery/'+mod+' - beta0.png', bbox_inches='tight', dpi=600)
    # Correlation beta1
    stats.pearsonr(modell_b1, choice_b1)
    sns_plot = sns.lmplot(data = beta1, x="Beta0 model", y="Beta1 target")
    plt.gcf().set_size_inches(6, 6)
    plt.tick_params(axis="x", labelsize=24)
    plt.tick_params(axis="y", labelsize=24)
    plt.title(mod, fontsize = 38)
    plt.ylabel(r"$\beta_1$ target", fontsize=30)
    plt.xlabel(r"$\beta_1$ model", fontsize=30)
    # sns_plot.figure.savefig('/home/sergej/Documents/academics/dnhi/projects/AAA/FORA02/RESULTS/figures_intern/parameter_recovery/'+mod+' - beta1.png', bbox_inches='tight', dpi=600)
    
    ## Recovered parameter distributions
    # Beta0
    fig, ax = plt.subplots(figsize=(6, 6), dpi=600)
    ax.hist(b0_subs[itr])
    ax.tick_params(axis="x", labelsize=24)
    ax.tick_params(axis="y", labelsize=24)
    plt.title(mod, fontsize = 38)
    plt.ylabel("Frequency", fontsize=30)
    plt.xlabel(r"$\beta_0$", fontsize=30)
    # Beta1
    fig, ax = plt.subplots(figsize=(6, 6), dpi=600)
    ax.hist(b1_subs[itr])
    ax.tick_params(axis="x", labelsize=24)
    ax.tick_params(axis="y", labelsize=24)
    plt.title(mod, fontsize = 38)
    plt.ylabel("Frequency", fontsize=30)
    plt.xlabel(r"$\beta_1$", fontsize=30)

## Plotting
# Prepare data
confusion_mat = pd.DataFrame(confusion_matrix, range(len(confusion_matrix)), range(len(confusion_matrix)))
confusion_mat.columns = mdlName
confusion_mat['index'] = mdlName
confusion_mat = confusion_mat.set_index('index')
# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 8), dpi=600)
cmap = sns.cm.rocket_r
ax = sns.heatmap(confusion_mat, annot=True, yticklabels=True, fmt=".2%",
                  xticklabels=True, annot_kws={'size': 24}, cmap = cmap)
ax.tick_params(axis="x", labelsize=24, labelrotation=70)
ax.tick_params(axis="y", labelsize=24, labelrotation=0)
plt.ylabel("True class", fontsize=30)
plt.xlabel("Predicted class", fontsize=30)
cax = ax.figure.axes[-1]
cax.tick_params(labelsize=24)

## Print coefficient distributions
beta0 = pd.DataFrame([mdlName, mu_b0, si_b0]).T
beta0.columns = ["model name", "beta0 mean", "beta0 std"]
print("============================================")
print("Intercepts distributions for selected models")
print("============================================")
print(beta0)
beta1 = pd.DataFrame([mdlName, mu_b1, si_b1]).T
beta1.columns = ["model name", "beta1 mean", "beta1 std"]
print("============================================")
print("Slope coef distributions for selected models")
print("============================================")
print(beta1)