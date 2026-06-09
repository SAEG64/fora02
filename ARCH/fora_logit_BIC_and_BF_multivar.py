#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:33:38 2023

@author: sergej
"""

# Requirements
import glob
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
# from scipy.stats import pearsonr
from patsy.contrasts import Treatment
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sympy import *
import math
from copy import deepcopy
import seaborn as sns
import os
path = os.path.dirname(__file__)+"/"

os.chdir(path)
# Select data subset for condition
# 1 for p success dominant heuristic data (loose competition)
# 2 for r threat encounter dominant data (tense competition)
# If anything else: whole dataset is selected
condition = 0

# Logit derivative for RT regression
x = symbols('x')
f = 1/(1+math.e**(-x))
fDiff = Derivative(f, x)

# List of model names
mdlName = [
    'win stay lose shift',
    # '** wait when safe',
    # '** binary energy state',
    'weather type',
    '* $\\mathit{r}$ predator',
    '* $\\mathit{p}$ gain',
    'expected gain naive',
    '** $\\mathit{p}$ success',
    # 'marginal value',
    # '$\\mathit{p}$ success + BES',
    # '$\\mathit{p}$ success + WWS',
    # 'multi-heuristic policy',
    # '$\mathit{OP}$ values + cap',
    'optimal policy values'
]
# if condition == 1 or condition == 2:
#     # List of model names
#     mdlName = [
#         'win stay lose shift',
#         # '** wait when safe',
#         # '** binary energy state',
#         'weather type',
#         '* $\\mathit{r}$ predator',
#         '* $\\mathit{p}$ gain',
#         'expected gain naive',
#         '** $\\mathit{p}$ success',
#         # 'marginal value',
#         # '$\\mathit{p}$ success + BES',
#         # '$\\mathit{p}$ success + WWS',
#         # 'multi-heuristic policy',
#         # '$\mathit{OP}$ values + cap',
#         # 'ternary state',
#         'optimal policy values'
#     ]

if __name__ == '__main__':
    wws_count = []
    bes_count = []
    # Logit for subject-level data
    bic_all = []  # BICs for all models and subjects
    datall = []   # All data concat
    s_count = 1
    for itr, fle in enumerate(glob.glob(path + "DATA_clean/test_data.*.CAT.csv")):
        # print(fle)
        # Get subject's data
        sbj = fle[len(path+"DATA_clean/test_data."):-len(".CAT.csv")]
        # print(sbj)
        dt = pd.read_csv(path + "DATA_clean/test_data." +
                         sbj + ".CAT" + ".csv")
        
        # Add ternary state model
        BNW_state = []
        for index, row in dt.iterrows():
            BNW_state.append(2)
            if row['** binary energy state'] == 1:
                BNW_state[index] = 1
            elif row['** wait when safe'] == 0:
                BNW_state[index] = 3
        dt['ternary state'] = BNW_state

        # Filter data
        # drop none responses
        dt = dt[dt['foraging T/F NaNs'].isnull() == False]
        # Select condition (if given)
        if condition == 1:
            dt = dt[dt["p/r heuristic"] == "['p']"]
        elif condition == 2:
            dt = dt[dt["p/r heuristic"] == "['r']"]
        dt = dt.reset_index(drop=True)
        
        # Add subject ID
        dt['ID_nr'] = s_count
        s_count += 1
        
        # Test BES and WWS fit
        # wws_count.append(len(dt[dt['** wait when safe'] == 0]))
        # bes_count.append(len(dt[dt['** binary energy state'] == 1]))

        # Regression for each model
        bic = []  # BIC per model
        dtC = pd.DataFrame({'A': np.arange(len(dt))})
        # Multivariate models
        for nme in mdlName[:]:
            
            print('===============================================')
            print(nme)
            print('===============================================')
            
            # Add multivariate
            m1_name = 'ternary state'
            m1 = list(dt[m1_name])
            
            # Prepare data
            # Get model
            m_raw = list(dt[nme])
            # Get reponses
            respo = list(np.array(dt["fora_response"]))
            # Correct data for eventual NaNs in model
            respo = [np.nan if np.isnan(m_raw[k]) else respo[k]
                      for k in range(0, len(m_raw))]
            model = [np.nan if np.isnan(respo[k]) else m_raw[k]
                      for k in range(0, len(respo))]
            m1 = [999 if np.isnan(respo[k]) else m1[k]
                      for k in range(0, len(respo))]
            model = [x for x in model if np.isnan(x) == False]
            m1 = [x for x in m1 if x != 999]
            respo = [x for x in respo if np.isnan(x) == False]
            respo = np.array(respo)
            model_copy = deepcopy(model)
            # Add constant for intercept
            model = np.array(model)
            # model = sm.add_constant(model)
            # m1 = sm.add_constant(m1)
            
            # Contrast code
            levels = [1,2,3]
            contrast = Treatment(reference=0).code_without_intercept(levels)
            c_mat = contrast.matrix[pd.DataFrame(m1)-1, :]
            c_mat = np.array(c_mat)
            
            # Check for multicollinearity
            print('===================================')
            print('correlation matrix')
            mds = pd.DataFrame(np.c_[ np.array(m1), model ])
            correlation_matrix = mds.corr()
            print(correlation_matrix)
            # # Visualize the correlation matrix
            # plt.figure(figsize=(10, 8))
            # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            # plt.title("Correlation Matrix")
            # plt.show()
            
            # Calculate VIF for each feature
            print('===================================')
            print('variance inflation factor')
            X = sm.add_constant(mds)
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            # Print the VIF values
            print(vif_data)


            # Run logit
            mod_fin = np.c_[ c_mat[:,0], model ]              # add a column
            mdl = sm.Logit(respo, sm.tools.tools.add_constant(mod_fin))
            # Fit with BFGS to handle singularity in design matrix
            exog = mdl.exog
            u, s, vt = np.linalg.svd(exog, 0)
            result = mdl.fit_regularized()
            # Append BIC value
            bic.append(result.bic)

            # Compute model uncertainties
            # response times
            rt = [np.nan if np.isnan(m_raw[i]) else dt.iloc[i]['logRT']
                  for i in range(0, len(m_raw[:]))]
            rt = [x for x in rt if np.isnan(x) == False]
            rt = np.array(rt)
            # Compute model derivative
            dv = [result.params[0] + result.params[1]*model_copy[i]
                  for i in range(0, len(model_copy))]
            uncer = [float(fDiff.doit().subs({x: dv[i]}))
                      for i in range(0, len(dv))]
            # Copy regressor
            uncertainty = deepcopy(uncer)
            # Run linear regression
            uncer = np.array(uncer)
            uncer = sm.add_constant(uncer)
            # Predict RT
            uc_mdl = sm.OLS(rt, uncer).fit()
            uc_result = uc_mdl.predict(uncer)

            # Expand data with fits and uncertainties
            if len(result.predict()) != len(dt):
                dtC[nme + ' fit'] = [np.nan] + list(result.predict())
                dtC[nme + '_uncertain'] = [np.nan] + uncertainty
                dtC[nme + '_uncertain_fit'] = [np.nan] + list(uc_result)
            else:
                dtC[nme + ' fit'] = result.predict()
                dtC[nme + '_uncertain'] = uncertainty
                dtC[nme + '_uncertain_fit'] = list(uc_result)
                
        # # Optimal policy univariate
        # for nme in mdlName[-1:]:
            
        #     # Prepare data
        #     # Get model
        #     m_raw = list(dt[nme])
        #     # Get reponses
        #     respo = list(np.array(dt["fora_response"]))
        #     # Correct data for eventual NaNs in model
        #     respo = [np.nan if np.isnan(m_raw[k]) else respo[k]
        #               for k in range(0, len(m_raw))]
        #     model = [np.nan if np.isnan(respo[k]) else m_raw[k]
        #               for k in range(0, len(respo))]
        #     model = [x for x in model if np.isnan(x) == False]
        #     respo = [x for x in respo if np.isnan(x) == False]
        #     respo = np.array(respo)
        #     model_copy = deepcopy(model)
        #     # Add constant for intercept
        #     model = np.array(model)
        #     model = sm.add_constant(model)

        #     # Run logit
        #     mdl = sm.Logit(respo, model)
        #     # Fit with BFGS to handle singularity in design matrix
        #     exog = mdl.exog
        #     u, s, vt = np.linalg.svd(exog, 0)
        #     result = mdl.fit()
        #     # Append BIC value
        #     bic.append(result.bic)

        #     # Compute model uncertainties
        #     # response times
        #     rt = [np.nan if np.isnan(m_raw[i]) else dt.iloc[i]['logRT']
        #           for i in range(0, len(m_raw[:]))]
        #     rt = [x for x in rt if np.isnan(x) == False]
        #     rt = np.array(rt)
        #     # Compute model derivative
        #     dv = [result.params[0] + result.params[1]*model_copy[i]
        #           for i in range(0, len(model_copy))]
        #     uncer = [float(fDiff.doit().subs({x: dv[i]}))
        #               for i in range(0, len(dv))]
        #     # Copy regressor
        #     uncertainty = deepcopy(uncer)
        #     # Run linear regression
        #     uncer = np.array(uncer)
        #     uncer = sm.add_constant(uncer)
        #     # Predict RT
        #     uc_mdl = sm.OLS(rt, uncer).fit()
        #     uc_result = uc_mdl.predict(uncer)

        #     # Expand data with fits and uncertainties
        #     if len(result.predict(model)) != len(dt):
        #         dtC[nme + ' fit'] = [np.nan] + list(result.predict(model))
        #         dtC[nme + '_uncertain'] = [np.nan] + uncertainty
        #         dtC[nme + '_uncertain_fit'] = [np.nan] + list(uc_result)
        #     else:
        #         dtC[nme + ' fit'] = result.predict(model)
        #         dtC[nme + '_uncertain'] = uncertainty
        #         dtC[nme + '_uncertain_fit'] = list(uc_result)
        
        # Concat fitted data
        del dtC['A']
        dt_REV = pd.concat([dt, dtC], axis=1).reset_index(drop=True)
        datall.append(dt_REV)
        
        # Append models' BICs and model coefficients
        bic_all.append(bic)

    # Compute log-group Bayes factor
    # Transpose subject-model to model-subject order
    bic_all = np.array(bic_all).T
    bcsums = []
    for i in range(0, len(bic_all)):
        bcsums.append(sum(bic_all[i]))
    # Log group Bayes Factor
    bcsums = bcsums-bcsums[-1]
    bcsums = pd.DataFrame(bcsums).T
    bcsums.columns = mdlName

    # Save BICs for PEP computation (with matlab script)
    bic_all = [pd.DataFrame(li) for li in bic_all]
    bicsRAW = pd.concat([pd.DataFrame(li)
                        for li in bic_all], axis=1).reset_index(drop=True)
    bicsRAW.columns = mdlName
    '''
    # ATTENTION: The following line exports the BIC values computed in the logit
    # regressions computed in this script. To run compute the protected exceedance
    # probabilities by your self, the BICs have to be overwritten with this script
    # for the condition you want to run the analysis for. After that, the 
    # fora_PEP_values.m script will compute the PEP values for the BIC values
    # exported in the following line.
    '''
    # bicsRAW.to_csv(path + 'RESULTS/fora_BICs.csv', index=False)
    
    mdlName = [
        'win stay lose shift',
        # 'wait when safe',
        # 'binary energy state',
        'weather type',
        '$\\mathit{r}$ predator',
        '$\\mathit{p}$ gain',
        'expected gain naive',
        '$\\mathit{p}$ success',
        # 'marginal value',
        # '$\\mathit{p}$ success + BES',
        # '$\\mathit{p}$ success + WWS',
        # 'multi-heuristic policy',
        # '$\mathit{OP}$ values + cap',
        'optimal policy values'
        ]
    # for i in range(len(mdlName)):
    #     mdlName[i] = 'ternary state + ' + mdlName[i]
    # Plot log-group Bayes factors
    if condition == 0:
        # Plotting
        name = mdlName
        valu = bcsums.values.tolist()[0]
        # Figure Size
        fig, ax = plt.subplots(figsize=(16, 8))
        # Increase x and y labels
        ax.tick_params(axis="x", labelsize=34)
        ax.tick_params(axis="y", labelsize=34)
        ax.tick_params(bottom=True, left=True, size=5, direction="in")
        # Horizontal Bar Plot
        ax.barh(name, valu)
        # ax.get_yticklabels()[-2].set_color("blue")
        # Add Plot Title
        ax.set_title('log group Bayes factor (BF)',
                      loc='left', size=46)
        plt.xlabel("BF (lower is better)", fontsize=40)
        # ax.autoscale(enable=True) 

    elif condition == 1:
        # Plotting p pool
        name = mdlName
        valu = bcsums.values.tolist()[0]
        # Figure Size
        fig, ax = plt.subplots(figsize=(16, 8))
        # Increase x and y labels
        ax.tick_params(axis="x", labelsize=34)
        ax.tick_params(axis="y", labelsize=34)
        ax.tick_params(bottom=True, left=True, size=5, direction="in")
        # Horizontal Bar Plot
        ax.barh(name, valu)
        # ax.get_yticklabels()[-3].set_color("blue")
        # Add Plot Title
        ax.set_title('BF for approach forests',
                      loc='left', size=46)
        plt.xlabel("BF (lower is better)", fontsize=40)

    elif condition == 2:
        # Plotting r pool
        name = mdlName
        valu = bcsums.values.tolist()[0]
        # Figure Size
        fig, ax = plt.subplots(figsize=(16, 8))
        # Increase x and y labels
        ax.tick_params(axis="x", labelsize=34)
        ax.tick_params(axis="y", labelsize=34)
        ax.tick_params(bottom=True, left=True, size=5, direction="in")
        # Horizontal Bar Plot
        ax.barh(name, valu)
        # ax.get_yticklabels()[-2].set_color("blue")
        # ax.get_yticklabels()[-3].set_color("blue")
        # Add Plot Title
        ax.set_title('BF for avoidance forests',
                      loc='left', size=46)
        plt.xlabel("BF (lower is better)", fontsize=40)

    # Test avoidance condition 'effect size'
    bics_T = np.array(bic_all).T
    wins = np.zeros((len(bics_T[0][0])))
    for i in range(len(bics_T[0])):
        # print(12-np.argmin(bics_T[0][i]))
        wins[len(bics_T[0][0])-1-np.argmin(bics_T[0][i])] += 1
    mdlName = [
        # '#12',
        # '#11',
        # '#10',
        # '#9',
        # '#8',
        '#7',
        '#6',
        '#5',
        '#4',
        '#3',
        '#2',
        '#1'
    ]
    if condition == 1 or condition == 2:
        mdlName = [
            # '#12',
            # '#11',
            # '#10',
            # '#9',
            # '#8',
            '#7',
            '#6',
            '#5',
            '#4',
            '#3',
            '#2',
            '#1'
        ]
    # Plotting r pool
    name = mdlName
    valu = wins[::-1]
    # Figure Size
    fig, ax = plt.subplots(figsize=(6, 8))
    # Increase x and y labels
    ax.tick_params(axis="x", labelsize=34)
    ax.tick_params(axis="y", labelsize=34)
    ax.tick_params(bottom=True, left=True, size=5, direction="in")
    # Horizontal Bar Plot
    ax.barh(name, valu)
    # ax.get_yticklabels()[-2].set_color("blue")
    # ax.get_yticklabels()[-3].set_color("blue")
    # Add Plot Title
    ax.set_title('   ',
                  loc='left', size=46)
    plt.xticks([0, 10, 20], ['0', '10', '20'])
    ax.set_title('$\mathit{n}$ model wins',
                  loc ='left', size = 46)
    # ax.autoscale(enable=True) 
    
    # Make concat data
    catData = pd.concat(datall, ignore_index = True)
    '''
    # Export all regression outputs per subject
    # ATTENTION: uncommenting the subsequent line will cause some data files to be
    # overwritten if analysis is only done for a subset (conditions) of the
    # data. This may affect the outcome of other scripts.
    '''
    # dt_REV.to_csv(path + "DATA_clean/DATA_fitted/test_data." + sbj + ".CAT" + "_regress" + ".csv", index = False)
    
    # p success data binning
    catData['resp_count'] = 1   # Variable to compute sampling rate
    catData = catData[['ID_nr','ternary state','** $\\mathit{p}$ success','fora_response','logRT','key_resp.rt','resp_count']]
    catData['p_succ_bin'] = pd.cut(catData['** $\\mathit{p}$ success'], bins=7, include_lowest=True)
    # Filtering
    catFilt = catData[np.isnan(catData['fora_response']) == False] # criteria applies by definition
    # # Handle any NaN values by filling them with a default label (e.g., 'Unknown')
    # df['p_succ_bin'] = df['p_succ_bin'].fillna('Unknown')
    
    # Investigate singularity issue with multivariate ternary state
    # Aggregated mean data
    aggrega = catFilt.groupby(['ID_nr','ternary state','p_succ_bin']).mean()
    aggrega = aggrega.reset_index()
    # Count occurances
    agg_sum = catFilt.groupby(['ID_nr','ternary state','p_succ_bin'])['resp_count'].sum()
    agg_sum = agg_sum.reset_index()
    # Get standard deviations and standard errors
    agg_BNW = catFilt.groupby(['ID_nr','ternary state','p_succ_bin'])['logRT'].std()
    agg_BNW = agg_BNW.reset_index()
    agg_BNW = agg_BNW.rename(columns = {'logRT': 'logRT_std'})
    agg_BNW['logRT_sem'] = agg_BNW['logRT_std']/agg_sum['resp_count']
    agg_BNW['RT_sem'] = np.exp(agg_BNW['logRT_sem'])
    agg_BNW['choice_sem'] = np.sqrt((aggrega['fora_response']*(1-aggrega['fora_response']))/agg_sum['resp_count'])
    agg_BNW['resp_count'] = agg_sum['resp_count']
    agg_BNW['choice_mean'] = aggrega['fora_response']
    agg_BNW['logRT'] = aggrega['logRT']
    agg_BNW['RT'] = aggrega['key_resp.rt']
    agg_BNW['** $\\mathit{p}$ success'] = aggrega['** $\\mathit{p}$ success']
    # Final averaging over all subjects
    agg_fin = agg_BNW.groupby(['ternary state','p_succ_bin']).mean()
    agg_fin = agg_fin.reset_index()
    
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'Arial'
    sns.set_style("white")
    sns.set_palette("Paired")

    # Plot condition splits for p success with respect to p foraging
    fig, ax = plt.subplots(figsize=(6, 6),dpi = 600)
    ci = agg_fin['choice_sem']*1.96 # confidence interval
    plot_pFora = sns.regplot(
        x='** $\\mathit{p}$ success', y='choice_mean', data=agg_fin, logistic=True, ci=None,
        ax=ax, label='', scatter_kws={'s':agg_fin['resp_count']*3}, 
        line_kws = {"color": "None"})
    ax.errorbar(
        x='** $\\mathit{p}$ success', y='choice_mean', data=agg_fin, yerr = ci, fmt='none', capsize=0, 
        zorder=1, color='C0', label=None)
    sns.lineplot(x='** $\\mathit{p}$ success', y='choice_mean', data=agg_fin,
            hue='ternary state', ax=ax)
    ax.tick_params(bottom=True, left=True, size=5, direction= "in")
    plt.ylabel("Foraging likelihood", fontsize=30)
    plt.xlabel("$\\mathit{p}$ success binned", fontsize=30)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    ax.tick_params(axis="x", labelsize=24)
    ax.tick_params(axis="y", labelsize=24)

    # Plot condition splits for p success with respect to RT
    fig, ax = plt.subplots(figsize=(6, 6),dpi = 600)
    ci = agg_fin['logRT_sem']*1.96 # confidence interval
    plot_pFora = sns.regplot(
        x='** $\\mathit{p}$ success', y='logRT', data=agg_fin, logistic=True, ci=None,
        ax=ax, label='', scatter_kws={'s':agg_fin['resp_count']*3}, 
        line_kws = {"color": "None"})
    ax.errorbar(
        x='** $\\mathit{p}$ success', y='logRT', data=agg_fin, yerr = ci, fmt='none', capsize=0, 
        zorder=1, color='C0', label=None)
    sns.lineplot(x='** $\\mathit{p}$ success', y='logRT', data=agg_fin,
            hue='ternary state', ax=ax)
    ax.tick_params(bottom=True, left=True, size=5, direction= "in")
    plt.ylabel("log(RT)", fontsize=30)
    plt.xlabel("$\\mathit{p}$ success binned", fontsize=30)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    ax.tick_params(axis="x", labelsize=24)
    ax.tick_params(axis="y", labelsize=24)
    # plt.ylim(ymin=6.3)
    
    
# =============================================================================
#   Run condition split and plotting for optimal policy
# =============================================================================
    
    catData = pd.concat(datall, ignore_index = True)
    catData['type'] = 'trade-off'
    catData['type'] = ['BES' if catData.iloc[i]['ternary state'] == 1 else catData.iloc[i]['type'] for i in range(len(catData))]
    catData['type'] = ['WWS' if catData.iloc[i]['ternary state'] == 3 else catData.iloc[i]['type'] for i in range(len(catData))]
    
    # Filtering
    # catData = catData[catData['ternary state'] == 2]
    # catData = catData[catData['p/r heuristic'] == "['r']"]
    catData['resp_count'] = 1   # Variable to compute sampling rate
    catData = catData[['ID_nr','ternary state', 'type', 'optimal policy values','fora_response','logRT','key_resp.rt','resp_count']]
    # catData['OP_bin'] = pd.cut(catData['optimal policy values'], bins=10, include_lowest=True)
    # Bin ternary state 2 sepparate
    cat_det = catData[catData['ternary state'] == 2]
    cat_det['OP_bin'] = pd.cut(cat_det['optimal policy values'], bins=7, include_lowest=True)
    catData = catData[catData['ternary state'] != 2]
    catData['OP_bin'] = pd.cut(catData['optimal policy values'], bins=7, include_lowest=True)
    catData = pd.concat([catData, cat_det])
    # Filtering
    catFilt = catData[np.isnan(catData['fora_response']) == False] # criteria applies by definition

    # Aggregated mean data
    aggrega = catFilt.groupby(['ID_nr','type','OP_bin']).mean()
    aggrega = aggrega.reset_index()
    # Count occurances
    agg_sum = catFilt.groupby(['ID_nr','type','OP_bin'])['resp_count'].sum()
    agg_sum = agg_sum.reset_index()
    # Get standard deviations and standard errors
    agg_BNW = catFilt.groupby(['ID_nr','type','OP_bin'])['logRT'].std()
    agg_BNW = agg_BNW.reset_index()
    agg_BNW = agg_BNW.rename(columns = {'logRT': 'logRT_std'})
    agg_BNW['logRT_sem'] = agg_BNW['logRT_std']/agg_sum['resp_count']
    agg_BNW['RT_sem'] = np.exp(agg_BNW['logRT_sem'])
    agg_BNW['choice_sem'] = np.sqrt((aggrega['fora_response']*(1-aggrega['fora_response']))/agg_sum['resp_count'])
    agg_BNW['resp_count'] = agg_sum['resp_count']
    agg_BNW['choice_mean'] = aggrega['fora_response']
    agg_BNW['logRT'] = aggrega['logRT']
    agg_BNW['RT'] = aggrega['key_resp.rt']
    agg_BNW['optimal policy values'] = aggrega['optimal policy values']
    # Final averaging over all subjects
    agg_fin = agg_BNW.groupby(['type','OP_bin']).mean()
    agg_fin = agg_fin.reset_index()
    # Filter nans
    agg_fin = agg_fin[np.isnan(agg_fin['optimal policy values']) == False]
    
    
    # Plot p foraging
    fig, ax = plt.subplots(figsize=(6, 6),dpi = 600)
    ci = agg_fin['choice_sem']*1.96 # confidence interval
    plot_pFora = sns.regplot(
        x='optimal policy values', y='choice_mean', data=agg_fin, logistic=True, ci=None,
        ax=ax, label='', scatter_kws={'s':agg_fin['resp_count']*3}, 
        line_kws = {"color": "None"})
    ax.errorbar(
        x='optimal policy values', y='choice_mean', data=agg_fin, yerr = ci, fmt='none', capsize=0, 
        zorder=1, color='C0', label=None)
    sns.lineplot(x='optimal policy values', y='choice_mean', data=agg_fin,
            hue='type', ax=ax)
    ax.tick_params(bottom=True, left=True, size=5, direction= "in")
    plt.ylabel("Foraging likelihood", fontsize=30)
    plt.xlabel("Optimal policy value bins", fontsize=30)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    ax.tick_params(axis="x", labelsize=24)
    ax.tick_params(axis="y", labelsize=24)

    # Plot RT
    fig, ax = plt.subplots(figsize=(6, 6),dpi = 600)
    ci = agg_fin['logRT_sem']*1.96 # confidence interval
    plot_pFora = sns.regplot(
        x='optimal policy values', y='logRT', data=agg_fin, logistic=True, ci=None,
        ax=ax, label='', scatter_kws={'s':agg_fin['resp_count']*3}, 
        line_kws = {"color": "None"})
    ax.errorbar(
        x='optimal policy values', y='logRT', data=agg_fin, yerr = ci, fmt='none', capsize=0, 
        zorder=1, color='C0', label=None)
    sns.lineplot(x='optimal policy values', y='logRT', data=agg_fin,
            hue='type', ax=ax)
    ax.tick_params(bottom=True, left=True, size=5, direction= "in")
    plt.ylabel("log(RT)", fontsize=30)
    plt.xlabel("Optimal policy value bins", fontsize=30)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    ax.tick_params(axis="x", labelsize=24)
    ax.tick_params(axis="y", labelsize=24)
    
    # Trial sampling
    catData = pd.concat(datall, ignore_index = True)
    # catData = catData[catData['ternary state'] == 3]
    # catData['OP_bin'] = pd.cut(catData['optimal policy values'], bins=14, include_lowest=True)
    # catData['binNr'] = [(
    #     float(str(catData.OP_bin[:][i]).replace(']','').replace('(','').split(',')[0])+
    #     float(str(catData.OP_bin[:][i]).replace(']','').replace('(','').split(',')[1]))/
    #     2 for i in range(len(catData))]
    fig, ax = plt.subplots(figsize=(6, 6),dpi = 600)
    ax.hist(catData['optimal policy values'])#, bins = list(catData['binNr']))
    # plt.ylim((0, epis/4))
    ax.tick_params(axis='x',which='major',labelsize=24)
    ax.tick_params(axis='y',which='major',labelsize=24)
    ax.set_title('WWS trials', fontsize = 38)
    ax.set_xlabel("Optimal policy value bins", fontsize = 30)
    ax.set_ylabel("Sampling frequency", fontsize = 30)
    
    # Check for correlations within trade-off choices
    catData = pd.concat(datall, ignore_index = True)
    catData = catData[catData['ternary state'] == 2]
    ops = pd.DataFrame(np.c_[list(catData['OP_value_difference']),list(catData['OP_value_difference_alternative'])])
    ops.corr()
    ops = pd.DataFrame(np.c_[list(catData['multi-heuristic policy']),list(catData['$\mathit{OP}$ values + cap'])])
    ops.corr()
