#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:22:35 2024

@author: sergej
"""

import pandas as pd
import glob
import os
path = os.path.dirname(__file__)+"/"
data = pd.read_csv('DATA_clean/DATA_group_level/test_data.group_level_datall.csv')
# data_subs = [data[data['participant'] == pd.unique(data['participant'])[i]] for i in range(len(pd.unique(data['participant'])))]

data_subs = []
for itr, fle in enumerate(glob.glob(path + "DATA_clean/test_data.*.CAT.csv")):
    # print(fle)
    # Get subject's data
    sbj = fle[len(path+"DATA_clean/test_data."):-len(".CAT.csv")]
    # print(sbj)
    dt = pd.read_csv(path + "DATA_clean/test_data." +
                     sbj + ".CAT" + ".csv")
    # Filter nans
    dt = dt[dt['foraging T/F NaNs'] != "None"]
    dt = dt.reset_index(drop=True)
    dt_cor = data[data['participant'] == dt.iloc[0]['Subject_ID']]
    if dt.iloc[0]['p/r heuristic'] == "['p']":
        dt_cor['cond_order'] = 'approach_first'
    else:
        dt_cor['cond_order'] = 'avoidance_first'
    data_subs.append(dt_cor)

data_REV = pd.concat(data_subs, axis = 0)
data_REV.to_csv(path + "DATA_clean/DATA_group_level/datall_with_condition_order.csv", index = False)