#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:33:38 2023

@author: sergej
"""

## Requirements
import glob
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sympy import *
import math
import statistics
from copy import deepcopy
import os
path = os.path.dirname(__file__)+"/"

os.chdir(path)
## Select data subset for condition
# 1 for p success dominant heuristic data (loose competition)
# 2 for r threat encounter dominant data (tense competition)
# If anything else: whole dataset is selected
condition = 0

## Logit derivative for RT regression
x = symbols('x')
f = 1/(1+math.e**(-x))
fDiff = Derivative(f, x)

# List of model names
mdlName =   [ 'weather type',
              'risk threat encounter',
              'win stay lose shift',
              '$\\mathit{p}$ success',
              'expected gain naive',
              'expected gain based MVT',
              '$\\mathit{p}$ success cor',
              'binary energy state',
              '$\\mathit{p}$ success cor + cap',
              'optimal policy']
# mdlName = model_names

if __name__ == '__main__':
    ## Logit for subject-level data
    bic_all = [] # BICs for all models and subjects
    for itr, fle in enumerate(glob.glob(path + "DATA_clean/test_data.*.CAT.csv")):
        # print(fle)
        ## Get subject's data
        sbj = fle[len(path+"DATA_clean/test_data."):-len(".CAT.csv")]
        # print(sbj)
        dt = pd.read_csv(path + "DATA_clean/test_data." + sbj + ".CAT" + ".csv")
        
        ## Filter data
        # drop none responses
        dt = dt[dt['foraging T/F NaNs'] != "None"]
        # Select condition (if given)
        if condition == 1:
            dt = dt[dt["p/r heuristic"] == "['p']"]
        elif condition == 2:
            dt = dt[dt["p/r heuristic"] == "['r']"]
        dt = dt.reset_index(drop=True)
        
        ## Regression for each model
        bic = [] # BIC per model
        dtC = pd.DataFrame({'A': np.arange(len(dt))})
        for nme in mdlName:
            
            ## Prepare data
            # Get model
            m_raw = dt[nme]
            # Get reponses
            respo = np.array(dt["fora_response"])
            # Correct data for eventual NaNs in model
            respo = [np.nan if np.isnan(m_raw[k]) else respo[k] for k in range(0, len(m_raw))]
            respo = [x for x in respo if np.isnan(x) == False]
            respo = np.array(respo)
            model = [x for x in m_raw if np.isnan(x) == False]
            model_copy = deepcopy(model)
            # Add constant for intercept
            model = np.array(model)
            model = sm.add_constant(model)
            
            ## Run logit
            mdl = sm.Logit(respo, model)
            # Fit with BFGS to handle singularity in design matrix
            exog = mdl.exog
            u, s, vt = np.linalg.svd(exog, 0)
            result = mdl.fit(method = "bfgs", maxiter = 100)
            # Append BIC value
            bic.append(result.bic)
            
            ## Compute model uncertainties
            # response times
            rt = [np.nan if np.isnan(m_raw[i]) else dt.iloc[i]['logRT'] for i in range(0, len(m_raw[:]))]
            rt = [x for x in rt if np.isnan(x) == False]
            rt = np.array(rt)
            # Compute model derivative
            dv = [result.params[0] + result.params[1]*model_copy[i] for i in range(0, len(model_copy))]
            uncer = [float(fDiff.doit().subs({x:dv[i]})) for i in range(0, len(dv))]
            # Copy regressor
            uncertainty = deepcopy(uncer)
            # Run linear regression
            uncer = np.array(uncer)
            uncer = sm.add_constant(uncer)
            # Predict RT
            uc_mdl = sm.OLS(rt,uncer).fit()
            uc_result = uc_mdl.predict(uncer)
            
            ## Expand data with fits and uncertainties
            if len(result.predict(model)) != len(dt):
                dtC[nme + ' fit'] = [np.nan] + list(result.predict(model))
                dtC[nme + '_uncertain'] = [np.nan] + uncertainty
                dtC[nme + '_uncertain_fit'] = [np.nan] + list(uc_result)
            else:
                dtC[nme + ' fit'] = result.predict(model)
                dtC[nme + '_uncertain'] = uncertainty
                dtC[nme + '_uncertain_fit'] = list(uc_result)
                
        # Concat fitted data
        del dtC['A']
        dt_REV = pd.concat([dt, dtC], axis = 1).reset_index(drop = True)
        
        # Append models' BICs and model coefficients
        bic_all.append(bic)
        
        # Export all regression outputs per subject
        # ATTENTION: uncommenting this line will cause some data files to be
        # overwritten if analysis is only done for a subset (conditions) of the
        # data. This may affect the outcome of other scripts.
        # dt_REV.to_csv(path + "DATA_clean/DATA_fitted/test_data." + sbj + ".CAT" + "_regress" + ".csv", index = False)
    
    ## Compute log-group Bayes factor
    bic_all = np.array(bic_all).T # Transpose subject-model to model-subject order 
    bcsums = []
    for i in range(0, len(bic_all)):
        bcsums.append(sum(bic_all[i]))
    # Log group Bayes Factor
    bcsums = bcsums-bcsums[-1]
    bcsums = pd.DataFrame(bcsums).T
    bcsums.columns = mdlName
    
    ## Save BICs for PEP computation (with matlab script)
    bic_all = [pd.DataFrame(li) for li in bic_all]
    bicsRAW = pd.concat([pd.DataFrame(li) for li in bic_all], axis=1).reset_index(drop=True)
    bicsRAW.columns = mdlName
    bicsRAW.to_csv(path + 'RESULTS/fora_BICs.csv', index = False)  
    
    ## Plot log-group Bayes factors
    if condition == 0:
        # Plotting
        name = mdlName
        valu = bcsums.values.tolist()[0]
        # Figure Size
        fig, ax = plt.subplots(figsize =(16, 8))
        # Increase x and y labels
        ax.tick_params(axis="x", labelsize=34)
        ax.tick_params(axis="y", labelsize=34)
        ax.tick_params(bottom=True, left=True, size=5, direction= "in")
        # Horizontal Bar Plot
        ax.barh(name, valu)
        ax.get_yticklabels()[-2].set_color("blue")
        # Add Plot Title
        ax.set_title('log group Bayes factor (BF)',
                      loc ='left', size = 46)
        plt.xlabel("BF (lower is better)", fontsize=40)
    
    elif condition == 1:
        # Plotting p pool
        name = mdlName
        valu = bcsums.values.tolist()[0]
        # Figure Size
        fig, ax = plt.subplots(figsize =(16, 8))
        # Increase x and y labels
        ax.tick_params(axis="x", labelsize=34)
        ax.tick_params(axis="y", labelsize=34)
        ax.tick_params(bottom=True, left=True, size=5, direction= "in")
        # Horizontal Bar Plot
        ax.barh(name, valu)
        ax.get_yticklabels()[-2].set_color("blue")
        # Add Plot Title
        ax.set_title('BF for loose competition forests',                 
                      loc ='left', size = 46)
        plt.xlabel("BF (lower is better)", fontsize=40)
    
    elif condition == 2:
        # Plotting r pool
        name = mdlName
        valu = bcsums.values.tolist()[0]
        # Figure Size
        fig, ax = plt.subplots(figsize =(16, 8))
        # Increase x and y labels
        ax.tick_params(axis="x", labelsize=34)
        ax.tick_params(axis="y", labelsize=34)
        ax.tick_params(bottom=True, left=True, size=5, direction= "in")
        # Horizontal Bar Plot
        ax.barh(name, valu)
        ax.get_yticklabels()[-2].set_color("blue")
        # Add Plot Title
        ax.set_title('BF for tense competition forests',
                      loc ='left', size = 46)
        plt.xlabel("BF (lower is better)", fontsize=40)