#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:06:51 2023

@author: sergej
"""

## Requirements
import glob
import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy.contrasts import Treatment
import statistics
import os
path = os.path.dirname(__file__)+"/"

## Beta coefficient distributions
def Extract(lst,x):
    lv = [item[x] for item in lst]
    return lv

## Select data subset for condition
# 1 for p success dominant heuristic data
# 2 for r threat encounter dominant data
# If anything else: whole dataset is selected
# condition = 2

# List of model names
mdlName = [
    # 'win stay lose shift',
    # '** wait when safe',
    # '** binary energy state',
    # 'weather type',
    'expected gain naive',
    '* $\\mathit{r}$ predator',
    '* $\\mathit{p}$ gain',
    '** $\\mathit{p}$ success',
    # 'marginal value',
    # '$\\mathit{p}$ success + BES',
    # '$\\mathit{p}$ success + WWS',
    'multi-heuristic policy',
    '$\mathit{OP}$ values + cap',
    'optimal policy values'
][::-1]

## Logit for subject-level data
def get_BIC():
    cof_all = [] ## Coefficients for models and subjects
    for itr, fle in enumerate(glob.glob(path + "DATA_clean/DATA_fitted/test_data.*.CAT_regress.csv")):
        # print(fle)
        ## Get subject's data
        sbj = fle[len(path+"DATA_clean/DATA_fitted/test_data."):-len(".CAT_regress.csv")]
        # print(sbj)
        dt = pd.read_csv(path + "DATA_clean/DATA_fitted/test_data." + sbj + ".CAT_regress" + ".csv")
        # Drop none responses
        dt = dt[dt['foraging T/F NaNs'] != "None"]
        dt = dt.reset_index(drop=True)
        # Modify OP energy 1 values
        
        ## Regression for each model
        cof = [] # Coefficients
        for nme in [mdlName[0]]:
            
            ## Prepare data
            # Get model
            m_raw = dt[nme]
            # Get reponses
            respo = np.array(dt["fora_response"])
            # Correct data for eventual NaNs in model
            respo = [np.nan if np.isnan(m_raw[i]) else respo[i] for i in range(0, len(m_raw[:]))]
            respo = [x for x in respo if np.isnan(x) == False]
            respo = np.array(respo)
            model = [x for x in m_raw if np.isnan(x) == False]
            # Add constant for intercept
            model = np.array(model)
            model = sm.add_constant(model)
            
            ## Run logit
            mdl = sm.Logit(respo, model)
            # Fit with BFGS to handle singularity in design matrix
            exog = mdl.exog
            u, s, vt = np.linalg.svd(exog, 0)
            result = mdl.fit(method = "bfgs", maxiter = 100)
            # Append coefficients
            cof.append(result.params)
        # # Append coefficients
        # cof_all.append(cof)
            
        # cof = [] # Coefficients
        for nme in mdlName[1:]:
            
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
            # Add constant for intercept
            model = np.array(model)
            # model = sm.add_constant(model)
            # m1 = sm.add_constant(m1)
            
            # Contrast code
            levels = [1,2,3]
            contrast = Treatment(reference=0).code_without_intercept(levels)
            c_mat = contrast.matrix[pd.DataFrame(m1)-1, :]
            c_mat = np.array(c_mat)
            
            # Run logit
            mod_fin = np.c_[ c_mat[:,0], model ]              # add a column
            mdl = sm.Logit(respo, sm.tools.tools.add_constant(mod_fin))
            # Fit with BFGS to handle singularity in design matrix
            exog = mdl.exog
            u, s, vt = np.linalg.svd(exog, 0)
            result = mdl.fit_regularized()
            # Append coefficients
            cof.append(result.params)
            # Append BIC value
            
        # Append coefficients
        cof_all.append(cof)
        
    b0_subs = [[] for i in range(len(mdlName))]
    b1_subs = [[] for i in range(len(mdlName))]
    b2_subs = [[] for i in range(len(mdlName))]
    b3_subs = [[] for i in range(len(mdlName))]
    for nr in range(0, len(cof_all)):
        # print(nr)
        # meanCof0 = np.mean(Extract(cofs, nr)[0])
        Cof0 = Extract(cof_all[nr], 0)
        Cof1 = Extract(cof_all[nr], 1)
        Cof2 = Extract(cof_all[nr][1:], 2)
        Cof3 = Extract(cof_all[nr][1:], 3)
        for el in range(len([mdlName[0]])):
            b0_subs[el].append(Cof0[el])
            b1_subs[el].append(Cof1[el])
        for el in range(len(mdlName)-1):
            ell = el+1
            b0_subs[ell].append(Cof0[ell])
            b1_subs[ell].append(Cof1[ell])
            b2_subs[ell].append(Cof2[el])
            b3_subs[ell].append(Cof3[el])
    # Filter coefs with content
    b2_subs = b2_subs[1:]
    b3_subs = b3_subs[1:]
    
    # print(b0_subs, b1_subs)
    mu_b0 = [np.mean(b0_subs[i]) for i in range(len(b0_subs))]
    mu_b1 = [np.mean(b1_subs[i]) for i in range(len(b1_subs))]
    mu_b2 = [np.mean(b2_subs[i]) for i in range(len(b2_subs))]
    mu_b3 = [np.mean(b3_subs[i]) for i in range(len(b3_subs))]
    min_b0 = [np.min(b0_subs[i]) for i in range(len(b0_subs))]
    max_b0 = [np.max(b0_subs[i]) for i in range(len(b0_subs))]
    min_b1 = [np.min(b1_subs[i]) for i in range(len(b1_subs))]
    max_b1 = [np.max(b1_subs[i]) for i in range(len(b1_subs))]
    min_b2 = [np.min(b2_subs[i]) for i in range(len(b2_subs))]
    max_b2 = [np.max(b2_subs[i]) for i in range(len(b2_subs))]
    min_b3 = [np.min(b3_subs[i]) for i in range(len(b3_subs))]
    max_b3 = [np.max(b3_subs[i]) for i in range(len(b3_subs))]
    si_b0 = [statistics.stdev(b0_subs[i]) for i in range(len(b0_subs))]
    si_b1 = [statistics.stdev(b1_subs[i]) for i in range(len(b1_subs))]
    si_b2 = [statistics.stdev(b2_subs[i]) for i in range(len(b2_subs))]
    si_b3 = [statistics.stdev(b3_subs[i]) for i in range(len(b3_subs))]
    
    return mdlName, mu_b0, mu_b1, mu_b2, mu_b3, si_b0, si_b1, si_b2, si_b3, b0_subs, b1_subs, b2_subs, b3_subs, cof_all, min_b0, max_b0, min_b1, max_b1, min_b2, max_b2, min_b3, max_b3

# mdlName, mu_b0, mu_b1, si_b0, si_b1, b0_subs, b1_subs = get_BIC(condition)
# beta_distributions = pd.DataFrame([mu_b0, mu_b1, si_b0, si_b1]).T
# idx = pd.Series(mdlName)
# hds = ["models", "beta0_mean", "beta1_mean", "beta0_std", "beta1_std"]
# beta_distributions = pd.concat([idx, beta_distributions], axis = 1)
# beta_distributions.columns = hds