#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:06:58 2023

@author: sergej
"""

import pandas as pd
import statsmodels.formula.api as smf
import os
path = os.path.dirname(__file__)+"/"

data = pd.read_csv(path + "DATA_clean/DATA_group_level/test_data.group_level.csv")

# Linear mixed effeccts models
md = smf.mixedlm("logRT ~ condition_rORp", data, groups=data["subject_ID"])
mdf = md.fit()
print(mdf.summary())
