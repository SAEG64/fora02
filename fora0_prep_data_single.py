# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import glob
import pandas as pd
import numpy as np
# import statsmodels.api as sm
import copy
# import matplotlib.pyplot as plt
from sympy import *
import math
import statistics
import os
path = os.path.dirname(__file__)+"/"
os.chdir(path)

"""==== Derrivative of logistic function ===="""
# Used for uncertainty of a model
x = symbols('x')
f = 1/(1+math.e**(-x))
fDiff = Derivative(f, x)

coefs = []
# Constants
n_blocks = 4
n_d = 8
# Table containing OP binary values for reindexing if data missing
mdpSheet = pd.read_csv(path + "/MDP_action_value_difference.csv", header=None, skiprows=1)

incl = []
cofs = []
mdlName = []
bicS = []
'''#######################################''''''
Subject and block loop for data extraction
''''''####################################'''
for itr, fle in enumerate(glob.glob(path + "raw_data/test_data.*.blck01.csv")):
    # print(itr)
    
    # Concat subjects' data blocks
    files = []
    for i in range(0, n_blocks):
        sbj = fle[len(path+"raw_data/test_data."):-len(".blck01.csv")]
        dat = pd.read_csv(path + "raw_data/test_data." + sbj + ".blck0" + str(i+1) + ".csv")
        files.append(dat)
    # print(sbj)
    dt = pd.concat(files, axis=0).reset_index(drop=True)
    dt = dt.drop(["welcoming.started", "welcoming.stopped", "key_resp_2.keys", "key_resp_2.rt", "key_resp_2.started", "key_resp_2.stopped", "Unnamed: 47", "n_day-1.out_LP.", "Unnamed: 48"], axis = 1)
    
    # Delete rows between trials
    dt.dropna(subset = ['day.thisN'], inplace = True)
    # Delete None response ("too late")
    dt['foraging T/F NaNs'] = dt['foraging T/F'].copy()
    dt['foraging T/F'] = dt['foraging T/F'].astype('str','float').map({'True': True, "TRUE": True, 'False': False, "FALSE": False, np.nan: 'None'}).astype(bool,float)
    dt = dt.reset_index(drop=True)

    '''###########################''''''
    Extract/correct data variables
    ''''''########################'''
    # Item sheets
    itemDays = []   # dayz
    for i in range(0, n_blocks):
        
        itms = pd.read_csv(path + "item_sheets/test_itemsDAYS." + sbj + ".blck" + str(i+1) + ".csv")
        itemDays.append(itms)
    itemDaysCat = pd.concat(itemDays, axis=0).reset_index(drop=True)
    itemsFora = pd.read_csv(path + "item_sheets/test_itemsFORA." + sbj + ".csv")  # forests
    
    dtC = copy.deepcopy(dt)
    # Reindex to OP values
    mvt_df = pd.DataFrame(columns=['forest_mean_p', 'mvt_cutoff_p', 'forest_mean_ev', 'mvt_cutoff_ev', 'mvt_policy_eg', 'mvt_policy_p'])
    for its, row in dt.iterrows():
        
        
        # Indexes between variable sheets
        i_day = row['day.thisN']
        lp_in = row['in_LP']
        itm_idx = int(row['miniblock.thisN']+int((row['session']-1)*18))
        day_idx = itemDaysCat.loc[itm_idx]['f_valsREV_rp_index_tracker']
        
        # "Real" variable side shown (unmirrored)
        siderz = [int(ii) for ii in itemDaysCat.loc[itm_idx]['env_side/day'][1:-1].split(',')]
        p_i = siderz[int(row['day.thisN'])]
        f_i = int(itemDaysCat.loc[itm_idx]['forest_flip'])

        # Optimal policy indexers:
        opr_idx = int(day_idx+1)*(n_d+1)-int(i_day)-1       # time points --> row
        if f_i ==0:
            opc_idx = int(lp_in + 7*p_i)                        # life points --> column
        else:
            opc_idx = int(lp_in + 7*(1-p_i))                    # life points --> column
        dtC.loc[its,'OP_value_difference'] = mdpSheet.iat[opr_idx,opc_idx]
        # Reassign heuristic values
        if p_i == 0 and f_i == 0:
            dtC.loc[its, 'p_succ_correct'] = itemsFora.loc[itm_idx]['pLeft_correct']
            dtC.loc[its, 'p_succ_uncorrected'] = itemsFora.loc[itm_idx]['pLeft_uncorrected']
            dtC.loc[its, 'r_threat'] = itemsFora.loc[itm_idx]['r_threatL']
            dtC.loc[its, 'n_oponents'] = itemsFora.loc[itm_idx]['n_oponentLeft']
            dtC.loc[its, 'gain_if_succ'] = itemsFora.loc[itm_idx]['value_succLeft']
            dtC.loc[its, 'weather_entropy'] = itemsFora.loc[itm_idx]['Shannon_H.L']
        elif p_i == 0 and f_i == 1:
            dtC.loc[its, 'p_succ_correct'] = itemsFora.loc[itm_idx]['pRight_correct']
            dtC.loc[its, 'p_succ_uncorrected'] = itemsFora.loc[itm_idx]['pRight_uncorrected']
            dtC.loc[its, 'r_threat'] = itemsFora.loc[itm_idx]['r_threatR']
            dtC.loc[its, 'n_oponents'] = itemsFora.loc[itm_idx]['n_oponentRight']
            dtC.loc[its, 'gain_if_succ'] = itemsFora.loc[itm_idx]['value_succRight']
            dtC.loc[its, 'weather_entropy'] = itemsFora.loc[itm_idx]['Shannon_H.R']
        elif p_i == 1 and f_i == 0:
            dtC.loc[its, 'p_succ_correct'] = itemsFora.loc[itm_idx]['pRight_correct']
            dtC.loc[its, 'p_succ_uncorrected'] = itemsFora.loc[itm_idx]['pRight_uncorrected']
            dtC.loc[its, 'r_threat'] = itemsFora.loc[itm_idx]['r_threatR']
            dtC.loc[its, 'n_oponents'] = itemsFora.loc[itm_idx]['n_oponentRight']
            dtC.loc[its, 'gain_if_succ'] = itemsFora.loc[itm_idx]['value_succRight']
            dtC.loc[its, 'weather_entropy'] = itemsFora.loc[itm_idx]['Shannon_H.R']  
        elif p_i == 1 and f_i == 1:
            dtC.loc[its, 'p_succ_correct'] = itemsFora.loc[itm_idx]['pLeft_correct']
            dtC.loc[its, 'p_succ_uncorrected'] = itemsFora.loc[itm_idx]['pLeft_uncorrected']
            dtC.loc[its, 'r_threat'] = itemsFora.loc[itm_idx]['r_threatL']
            dtC.loc[its, 'n_oponents'] = itemsFora.loc[itm_idx]['n_oponentLeft']
            dtC.loc[its, 'gain_if_succ'] = itemsFora.loc[itm_idx]['value_succLeft']
            dtC.loc[its, 'weather_entropy'] = itemsFora.loc[itm_idx]['Shannon_H.L']   
        
        dtC.loc[its, 'fora_NR'] = day_idx
       
        ## Compute values for marginal value theorem
        # Cutoff by cumulative mean
        if dtC.loc[its, 'day.thisRepN'] == 0:
            egLeft = itemsFora.loc[itm_idx]['pLeft_correct']*dtC.loc[its,'gain_if_succ'] + (1-itemsFora.loc[itm_idx]['pLeft_uncorrected'])*(1-itemsFora.loc[itm_idx]['r_threatL'])*-2 + itemsFora.loc[itm_idx]['r_threatL']*-3
            egRight = itemsFora.loc[itm_idx]['pRight_correct']*dtC.loc[its,'gain_if_succ'] + (1-itemsFora.loc[itm_idx]['pRight_uncorrected'])*(1-itemsFora.loc[itm_idx]['r_threatR'])*-2 + itemsFora.loc[itm_idx]['r_threatR']*-3
            p_cutoff = (itemsFora.loc[itm_idx]['pLeft_uncorrected'] + itemsFora.loc[itm_idx]['pRight_uncorrected'])/2
            ev_cutoff = (egLeft + egRight)/2
            if its > 0:
                p_cutoff = (mvt_df.loc[its-1, 'forest_mean_p'] + p_cutoff)/2
                ev_cutoff = (mvt_df.loc[its-1, 'forest_mean_ev'] + ev_cutoff)/2
        mvt_df.loc[its, 'mvt_cutoff_p'] = p_cutoff
        mvt_df.loc[its, 'mvt_cutoff_ev'] = ev_cutoff
        mvt_df.loc[its, 'forest_mean_p'] = (itemsFora.loc[itm_idx]['pLeft_uncorrected'] + itemsFora.loc[itm_idx]['pRight_uncorrected'])/2
        mvt_df.loc[its, 'forest_mean_ev'] = (egLeft + egRight)/2
        # Binary policy by cutoff
        if dtC.loc[its, 'p_succ_correct'] > mvt_df.loc[its, 'mvt_cutoff_p']:
            mvt_df.loc[its, 'mvt_policy_p'] = 1
        else:
            mvt_df.loc[its, 'mvt_policy_p'] = 0
        if (dtC.loc[its]['p_succ_correct']*dtC.loc[its]['gain_if_succ'] + (1-dtC.loc[its]['p_succ_uncorrected'])*(1-dtC.loc[its]['r_threat'])*-2 + dtC.loc[its]['r_threat']*-3) > mvt_df.loc[its, 'mvt_cutoff_ev']:
            mvt_df.loc[its, 'mvt_policy_eg'] = 1
        else:
            mvt_df.loc[its, 'mvt_policy_eg'] = 0
    dtC = pd.concat([dtC, mvt_df], axis = 1)
    
    # # # Subset selection
    anaGroup = 0
    if anaGroup != 0:
        # dtC = dtC[dtC["in_LP"] < 8-dtC["day.thisN"] + 1] # exclude save trials
        # dtC = dtC[dtC["in_LP"] > 1] # exclude binary energy state
        if anaGroup == 1:
            dtC = dtC[dtC["p/r heuristic"] == "['p']"]
        elif anaGroup == 2:
            dtC = dtC[dtC["p/r heuristic"] == "['r']"]
    dtC = dtC.reset_index(drop=True)
    
    
    '''############''''''
    Generate models
    ''''''#########'''
    ## Response variable
    resp = []
    # Loop through clean data
    for its, row in dtC.iterrows():
        # Convert fora bool (some are str, some are logic) to numeric
        if row['foraging T/F NaNs'] == "True":
            resp.append(1)
        elif row['foraging T/F NaNs'] == True:
            resp.append(1)
        elif row['foraging T/F NaNs'] == "False":
            resp.append(0)
        elif row['foraging T/F NaNs'] == False:
            resp.append(0)
        elif np.isnan(row['foraging T/F NaNs']) == True:
            resp.append(np.nan)
    resp = np.array(resp)   # resp
    dtC["fora_response"] = resp
    
    # Check if model separation error
    if np.all(resp) == False and len(resp)/8 < len(resp) - np.count_nonzero(resp == 1) < len(resp) - len(resp)/8:
        
        incl.append(itr)    # Check which subjects included
        
        ## log RTs
        dtC["logRT"] = [math.log(dtC.iloc[i]["key_resp.rt"]) for i in range(len(dtC["key_resp.rt"]))]
        
        ## Models
        """ Optimal policy """
        model = np.array(dtC['OP_value_difference'])
        # Add model to datasheet
        dtC["optimal policy values"] = model
        mdlName.append("optimal policy values")
        
        """ Heuristics """
        # p succ corrected - capped
        model = np.array(dtC['p_succ_correct'])
        for its, row in dtC.iterrows():
            if row['in_LP'] == 1:
                model[its] = 1
            elif row['in_LP'] > 8 - row["day.thisN"]:
                model[its] = 0
        # Add model to datasheet
        dtC["multi-heuristic policy"] = model
        mdlName.append("multi-heuristic policy")
        
        # p succ corrected
        model = np.array(dtC['p_succ_correct'])
        # Add model to datasheet
        dtC["** $\mathit{p}$ success"] = model
        mdlName.append("* $\mathit{p}$ success")
        
        # p succ uncorrected heuristic
        model = np.array(dtC['p_succ_uncorrected'])
        # Add model to datasheet
        dtC["* $\mathit{p}$ gain"] = model
        mdlName.append("$\mathit{p}$ gain")
        
        # "safe" from threat heuristic (1-r)
        model = dtC['r_threat']
        # Add model to datasheet
        dtC["* $\\mathit{r}$ predator"] = model
        mdlName.append("* $\\mathit{r}$ predator")
        
        # Expected gain p_UNC - uncapped
        model = np.array(dtC['p_succ_uncorrected']*dtC['gain_if_succ'] + (1-dtC['p_succ_uncorrected'])*-2 + dtC['r_threat']*-3) - (-1)
        # Add model to datasheet
        dtC["expected gain naive"] = model
        mdlName.append("expected gain naive")
        
        # Expected value
        model = np.array(dtC['p_succ_correct']*dtC['gain_if_succ'] + (1-dtC['p_succ_uncorrected'])*(1-dtC['r_threat'])*-2 + dtC['r_threat']*-3) - (-1)
        # Add model to datasheet
        dtC["expected gain correct"] = model
        mdlName.append("expected gain correct")
        
        # Expected value
        model = np.array(dtC['p_succ_correct']*dtC['gain_if_succ'] + (1-dtC['p_succ_uncorrected'])*(1-dtC['r_threat'])*-2 + dtC['r_threat']*-3) - (-1)
        model = model + dtC['in_LP']
        # Add model to datasheet
        dtC["expected state no bounds"] = model
        mdlName.append("expected state no bounds")        
        
        # Expected value p_COR - capped
        model = np.array(dtC['p_succ_correct']*dtC['gain_if_succ'] + (1-dtC['p_succ_uncorrected'])*(1-dtC['r_threat'])*-2 + dtC['r_threat']*-3)
        for j in range(0, len(model)):
            if dtC.iloc[j]['in_LP']+model[j] > 6:
                model[j] = 6 - dtC.iloc[j]['in_LP']
            elif dtC.iloc[j]['in_LP']+model[j] < 0:
                model[j] = -dtC.iloc[j]['in_LP']
        model = model - (-1)
        model = model + dtC['in_LP']
        # Add model to datasheet
        dtC["expected state with bounds"] = model
        mdlName.append("expected state with bounds")
        
        # Weather type
        model = np.multiply(np.array(dtC['good_weather.T/F']), 1).astype(float)
        # Add model to datasheet
        dtC["weather type"] = model
        mdlName.append("weather type")
        
        # simple gain model
        model = np.array(dtC['gain_if_succ'])
        # Add model to datasheet
        dtC["gain magnitude"] = model
        mdlName.append("gain magnitude")
            
        # Win-stay-lose-shift - binarized
        model = [1 if dtC.iloc[t]['out_LP']-dtC.iloc[t]['in_LP'] > 0 else -1 for t in range(0, len(dtC))][:-1]
        model = [np.nan] + model
        dtC["win stay lose shift"] = model # Add model to data sheet
        mdlName.append("win stay lose shift")
        
        """ Energy state heuristics """
        # Continuous energy state
        model = [dtC.iloc[i]['in_LP'] for i in range(0, len(dtC))]
        dtC["continuous energy state"] = model # Add model to data sheet
        mdlName.append("continuous energy state")
    
        # Binarized energy state
        model = [1 if dtC.iloc[n]['in_LP'] == 1 else 0 for n in range(0, len(dtC))]
        dtC["** binary energy state"] = model # Add model to data sheet
        mdlName.append("* binary energy state")
        
        # Wait-when-safe model
        model = [0 if dtC.iloc[t]['in_LP'] > (8 - dtC.iloc[t]['day.thisRepN']) else 1 for t in range(0, len(dtC))]
        dtC["** wait when safe"] = model # Add model to data sheet
        mdlName.append("* wait when safe")
        
        # MVT p policy
        model = np.array(dtC['mvt_policy_p'])
        dtC["$\mathit{p}$ success based MVT"] = model # Add model to data sheet
        mdlName.append("$\mathit{p}$ success based MVT")
        
        # MVT EG policy
        model = np.array(dtC['mvt_policy_eg'])
        dtC["marginal value"] = model # Add model to data sheet
        mdlName.append("marginal value")
        
        """ Additional models """
        # Capped optimal policy values
        """ Optimal policy + cap """
        model = [dtC.iloc[i]['OP_value_difference'] if dtC.iloc[i]['** binary energy state'] != 1 else 0.625 for i in range(len(dtC))]
        model = [model[i] if dtC.iloc[i]['** wait when safe'] != 0 else -0.84 for i in range(len(dtC))]
        # Add model to datasheet
        dtC["$\mathit{OP}$ values + cap"] = model
        mdlName.append("$\mathit{OP}$ values + cap")
        
        '''########################################################''''''
        Run models and extract Bayesian Information Criterion (BIC)
        ''''''#####################################################'''
        md_cofs = []
        bic = [] # List of subject's BIC values per model
        # Data frame for fitting and uncertaintyvectors
        fitSet = pd.DataFrame({'A': np.arange(len(dtC))})
        # Concat
        del fitSet['A']
        dtCREV = pd.concat([dtC, fitSet], axis = 1).reset_index(drop = True)
        
        # BIC values for all models per subject
        bicS.append(bic)
        cofs.append(md_cofs)
        
        # Export data with all model columns
        dataCorr = dtCREV.to_csv("DATA_clean/test_data." + sbj + ".CAT" + ".csv", index = False)
    else:
        skip = True
# incl = [*set(incl)]
# print("N =", len(incl))

mdlName = list(dict.fromkeys(mdlName))
model_names = mdlName


