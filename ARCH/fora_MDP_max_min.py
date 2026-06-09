#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:37:34 2024

@author: sergej
"""

import pandas as pd
import numpy as np
import os
path = os.path.dirname(__file__)+"/"
os.chdir(path)
data = pd.read_csv(path + "MDP_action_value_difference.csv")
np.min(data)
np.max(data)