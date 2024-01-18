# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import glob
import pandas as pd
import os
# import os
path = os.path.dirname(__file__)+"/"
# os.chdir("/home/sergej/Documents/academics/dnhi/projects/FORA/FORA028t/ANA/CLEAN/scripts/")
# from subject_level_BIC_and_lgGrpBF import condition

# Create empty pandas data frame
d = pd.read_csv(path + "DATA_clean/DATA_fitted/test_data.ae11.CAT_regress.csv")
catData = pd.DataFrame(columns = list(d.columns.values))

# Extract subject data and concat stacked frame
for itr, fle in enumerate(glob.glob(path + "DATA_clean/DATA_fitted/test_data.*.CAT_regress.csv")):
    # Get data and add subject ID
    sbj = fle[len(path+"DATA_clean/DATA_fitted/test_data."):-len(".CAT_regress.csv")]
    dt = pd.read_csv(path + "DATA_clean/DATA_fitted/test_data." + sbj + ".CAT_regress.csv")
    dt['subject_ID'] = itr+1
    # append catData sheet
    catData = catData.append(dt, ignore_index = True)

# Rename some variables for compatibility in R
catData['condition_rORp'] = catData['p/r heuristic'].astype('str').map({"['p']": "low threat condition", "['r']": "high threat condition"}).astype(str)
catData['expected_gain'] = catData['expected gain naive']
catData.to_csv(path + 'DATA_clean/DATA_group_level/test_data.group_level.csv')  


