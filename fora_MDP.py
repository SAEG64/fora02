#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:36:26 2024

@author: sergej
"""

import os
import array
import numpy as np
import pandas as pd
from copy import deepcopy
path = os.path.dirname(__file__)+"/"

# load forests
vars_t =  pd.read_csv(path + 'test_items.csv')

"""## Array variables ##"""
terminal = array.array("i", (-1, 0))            # Terminal states
n_fields = [2,3,4,5,6,7,8,9,10]                 # Ppossible grid sizes
n_foods = array.array("i", (1,2,3,4,5,6,7,8,9)) # fields containing food
values_gain = array.array("i", (1, 2))          # gains for fora succ
values_lossT = np.array((-3, 0))                # losses for threat event
n_oponent = [array.array("i", (0, 1, 2)),
                        array.array("i", (0, 1, 2))]   # number of predators in the game
"""## Integer variables ##""" 
p_waitn = 1  # probability fix choice (waiting)
lossFIX = -2 # loss for unsuccessful foraging
waitFIX = -1 # cost for waiting
n_states = 7    # number of states
n_d = 8         # number od time points in environment
maxLP = 6       # max LP state
p_env = 0.5     # probability of incurring environment
n_env = 2       # number of environments

"""## Get transition matrix ##""" 
def trans_mat(mat_raw, f_vals):
    
    trans_m = [[],[],[]]    # 2 foraging, 1 waiting lists for transitions per forest
    ## loop through items in f_vals (permutations sheet):
    for itm, row in f_vals.iterrows():
        
        # required variables:
        cost = row['value_nosucc']      # transition value unsuccess
        winL = row['value_succLeft']    # transition value success left env
        winR = row['value_succRight']   # transition value success right env
        thre = row['value_threat']      # transition value threat encounter
        wait = row['value_waiting']     # transition walue waiting
        pL_UNC = row['pLeft_uncorrected']       # p success left env uncorrected
        pR_UNC = row['pRight_uncorrected']      # p success right env uncorrected
        pL_COR = row['pLeft_correct']           # p success left env corrected
        pR_COR = row['pRight_correct']          # p success right env corrected
        #######################################################################
        p_thrL = row['r_threatL']      # p threat envounter left env
        p_thrR = row['r_threatR']      # p threat encounter right env
        #######################################################################
        trans_fL = deepcopy(mat_raw) # transformation matrix foraging left env raw
        trans_fR = deepcopy(mat_raw) # transformation matrix foraging right env raw
        trans_w = deepcopy(mat_raw)  # transformation matrix waiting (env independent)
        ## loop trough possible states of the game:
        for j in range(0,n_d):
            
            # Current LP state
            LP = j
            if LP >= 6:
                LP = 6
            elif LP <=0:
                LP = 0
            ## Indexes from capped transition values:
            """ 
            Indexes to assign transition probabilities are generated in regard to
            transition values for success, no success, threat encounter and waiting.
            """
            # loss transition index for no success:
            drain = LP+cost
            d = int()
            if drain <= -1:
                d = 0
            else:
                d = drain
            # gain transition index for success left env:
            gainL = LP+winL
            gL = int()
            if gainL >= maxLP:
                gL = int(maxLP)
            else:
                gL = gainL
            # gain transition index for success right env:
            gainR = LP+winR
            gR = int()
            if gainR >= maxLP:
                gR = int(maxLP)
            else:
                gR = gainR
            """
            Indexes for gL and gR are separated because trans mags were initially different.
            Could be changed to connstant and there would be no difference.
            """
            # loss transition if threat encounter:
            threat = LP+thre
            c = int()
            if threat <= -1:
                c = 0
            #elif threat >= n_states+1:
            #    c = n_states-1
            else:
                c = threat
            # waiting transition index:
            fastn = LP+wait
            w = int()
            if fastn<= -1:
                w = 0
            else:
                w = fastn
            ## Assign transition probabilities to respective indexes:
            if thre != 0:
                # env left:
                trans_fL[int(j)][int(gL)] = pL_COR             # gain transition
                trans_fL[int(j)][int(d)] = (1-pL_UNC)*(1-p_thrL)         # loss transition
                trans_fL[int(j)][int(c)] = (trans_fL[int(j)][int(c)]+p_thrL)   # additional loss transition
                # env right:
                trans_fR[int(j)][int(gR)] = pR_COR            # gain transition
                trans_fR[int(j)][int(d)] = (1-pR_UNC)*(1-p_thrR)         # loss transition
                trans_fR[int(j)][int(c)] = (trans_fR[int(j)][int(c)]+p_thrR)   # additional loss transition
            else:
                # env left:
                trans_fL[int(j)][int(gL)] = pL_UNC      # gain transition
                trans_fL[int(j)][int(d)] = 1-pL_UNC     # loss transition
                # env right:
                trans_fR[int(j)][int(gR)] = pR_UNC      # gain transition
                trans_fR[int(j)][int(d)] = 1-pR_UNC     # loss transition 
            trans_w[int(j)][int(w)] = p_waitn   # waiting (always fix)
        ## overwriting absorbing zero/death state:
        trans_fL[0][:] = 0
        trans_fL[0][0] = 1
        trans_fR[0][:] = 0
        trans_fR[0][0] = 1
        # append transitions to option lists
        trans_m[0].append(trans_fL)
        trans_m[1].append(trans_fR)
        trans_m[2].append(trans_w)
    # brake unnecessary list nest:
    trans_m = [np.array(i) for i in trans_m]
    return(trans_m)

'''## Run computation for transition matrices for all forests ##'''
mat_raw = np.zeros((n_d+1,n_states))
trans_mREV = trans_mat(mat_raw, vars_t)

# reformating  waiting and foraging arrays:
fL = [np.concatenate(trans_mREV[0]), np.concatenate(trans_mREV[2])]
fL[0] = np.concatenate((fL[0], fL[0]), axis = 1)*p_env
fL[1] = np.concatenate((fL[1], fL[1]), axis = 1)*p_env
fL = np.concatenate((fL[0], fL[1]), axis = 1)
fL = np.split((fL), len(vars_t))
fR = [np.concatenate(trans_mREV[1]), np.concatenate(trans_mREV[2])]
fR[0] = np.concatenate((fR[0], fR[0]), axis = 1)*p_env
fR[1] = np.concatenate((fR[1], fR[1]), axis = 1)*p_env
fR = np.concatenate((fR[0], fR[1]), axis = 1)
fR = np.split((fR), len(vars_t))
# final array in desired shape for MDP
trans_R2 = np.array([[fL[i],fR[i]] for i in range(0, len(vars_t))]) # final: nested transitions array

"""## MDP loop ##"""
rew_foraALL2 = []
rew_waitALL2 = []
pol_matALL2 = []
    
for n, i_for in enumerate(trans_R2):
    # define variables for computing rewards:
    pol_mat  = np.empty((n_d,n_states*n_env))
    pol_mat[:,:]      = np.nan
    rew_mat           = np.zeros((n_d+1,n_states*n_env))
    rew_mat[:,0]           = -1
    rew_mat[:,n_states]    = -1
    rew_wait = deepcopy(rew_mat)
    rew_fora = deepcopy(rew_mat)
    rew_cur  = np.zeros(2)
    
    for i_day in range(0, n_d):
        
        for nn, i_sta in enumerate(range(1, n_states)):
            
            for i_env in range(0, n_env):
                # transition vectors:
                trans_wait = i_for[i_env, i_sta,n_states*n_env:n_states*n_env*n_env] # waiting
                trans_fora = i_for[i_env, i_sta,:n_states*n_env]                     # foraging
                # current rewards:
                rew_cur[0] = np.dot(trans_wait,rew_mat[i_day])
                rew_cur[1] = np.dot(trans_fora,rew_mat[i_day])
                
                # policy matrix
                rew_arg = np.argmax(rew_cur)        # 0: wait; 1: forage
                if rew_cur[0] == rew_cur[1]:
                    rew_arg = 2                     # 2: indifference
                pol_mat[i_day, i_sta+i_env*n_states] = rew_arg
                # reward matrix
                rew_max = np.max(rew_cur)
                rew_mat[i_day+1, i_sta+i_env*n_states]   = rew_mat[i_day+1, i_sta+i_env*n_states]+rew_max 
                rew_wait[i_day+1, i_sta+i_env*n_states]   = rew_wait[i_day+1, i_sta+i_env*n_states]+rew_cur[0]
                rew_fora[i_day+1, i_sta+i_env*n_states]   = rew_fora[i_day+1, i_sta+i_env*n_states]+rew_cur[1]
        
    pol_matALL2.append(pol_mat)
    rew_waitALL2.append(rew_wait)
    rew_foraALL2.append(rew_fora)

'''## Formatting and export ##'''
# MDP waiting values:
rewWAIT = pd.DataFrame(np.concatenate(rew_waitALL2))
names = ['waitenv' + '1' for i in range(0, n_states)]+['waitenv' + '2' for j in range(0, n_states)]
rewWAIT.columns = names
# MDP foraging values
rewFORA = pd.DataFrame(np.concatenate(rew_foraALL2))
names = ['foraenv' + '1' for i in range(0, n_states)]+['foraenv' + '2' for j in range(0, n_states)]
rewFORA.columns = names
# MDP policy table
pol_REV = pd.DataFrame(np.concatenate(pol_matALL2))
names = ['env' + '1' for i in range(0, n_states)]+['env' + '2' for j in range(0, n_states)]
pol_REV.columns = names           
# MDP reward table
df_reward = pd.concat([rewFORA,rewWAIT], axis=1)
# Binarized difference foraging-waiting
op_value = pd.DataFrame(np.concatenate(rew_foraALL2)-np.concatenate(rew_waitALL2))
names = ['env' + '1' for i in range(0, n_states)]+['env' + '2' for j in range(0, n_states)]
op_value.columns = names
# Final products
MDP_rew_sheet = df_reward.to_csv(path + 'MDP_action_values.csv', index = False)
policy_sheet = pol_REV.to_csv(path + 'MDP_policy.csv', index = False)
op_val_sheet = op_value.to_csv(path + 'MDP_action_value_difference.csv', index = False)

''' 
    EXPLANATION: HOW TO GET THE RIGHT MDP VALUES
    Due to randomization of the forests, the order of participants' forests
    does not correspond with the order of the MDP spread sheets. Therefore,
    the values have to be extracted by correctly indexing into the state-action
    value differences. In the data sheets of each subjects, there is one column
    called fora_Nr, which represents the Nr of the forest in the order of the
    products of the MDP script. The correct MDP value can then be extracted
    from the MDP_action_value_difference sheet by applying the following function:
        
        int(fora_Nr+1)*(8+1)-int(day.thisN)-1
    
    where day.thisN is a variable representing time-points past in the data sheet.
'''
        
