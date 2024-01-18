#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:22:08 2023

@author: sergej
"""

# Requirements
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
path = os.path.dirname(__file__)+"/"


coefs_cond1 = [[],[]]
coefs_cond2 = [[],[]]
for itr, fle in enumerate(glob.glob(path + "DATA_clean/DATA_fitted/test_data.*.CAT_regress.csv")):
    # Extract participant's data
    sbj = fle[len(path+"DATA_clean/DATA_fitted/test_data."):-len(".CAT_regress.csv")]
    data = pd.read_csv(path + "DATA_clean/DATA_fitted/test_data." + sbj + ".CAT_regress.csv")
    # Separate Conditionss
    data_cond1 = data[data['p/r heuristic'] == "['p']"]
    data_cond2 = data[data['p/r heuristic'] == "['r']"]
    
    ## Get coefs for Conditions1
    # Get model
    model = data_cond1['risk threat encounter']
    # Get reponses
    respo = np.array(data_cond1["fora_response"])
    # Add constant for intercept
    model = np.array(model)
    model = sm.add_constant(model)
    # Run logit
    mdl = sm.Logit(respo, model)
    result = mdl.fit()
    # Append model coefscoefs_cond1
    coefs_cond1[0].append(result.params[0])
    coefs_cond1[1].append(result.params[1])
    
    ## Get coefs for Conditions2
    # Get model
    model = data_cond2['risk threat encounter']
    # Get reponses
    respo = np.array(data_cond2["fora_response"])
    
    model = np.array(model)
    model = sm.add_constant(model)
    # Run logit
    mdl = sm.Logit(respo, model)
    result = mdl.fit()
    # Append model coefscoefs_cond1
    coefs_cond2[0].append(result.params[0])
    coefs_cond2[1].append(result.params[1])

# Prepare data for plotting
df11 = pd.DataFrame(coefs_cond1[0])
df11['Coefficients'] = r"$\beta_0$"
df12 = pd.DataFrame(coefs_cond1[1])
df12['Coefficients'] = r"$\beta_1$"
df3 = pd.concat([df11, df12], axis = 0)
df3['Conditions'] = "loose"
df21 = pd.DataFrame(coefs_cond2[0])
df21['Coefficients'] = r"$\beta_0$"
df22 = pd.DataFrame(coefs_cond2[1])
df22['Coefficients'] = r"$\beta_1$"
df4 = pd.concat([df21, df22], axis = 0)
df4['Conditions'] = "tense"
df_stack = pd.concat([df3, df4], axis = 0).reset_index(drop=True)
df_stack = df_stack.rename(columns = {0:'β distributions'})
df_beta0 = df_stack[df_stack['Coefficients'] == r"$\beta_0$"]
df_beta0 = df_beta0.reset_index(drop=True)
df_beta1 = df_stack[df_stack['Coefficients'] == r"$\beta_1$"]
df_beta1 = df_beta1.reset_index(drop=True)

## Plotting
# Set publication level params for plotting
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
sns.set_style("white")
sns.set_palette("Paired")
# Make boxplot
fig, ax = plt.subplots(figsize=(6, 6), dpi = 600)
ax = sns.boxplot(x="Conditions", y="β distributions",
            hue="Coefficients", palette=["m", "g"],
            data=df_stack, showcaps=False, boxprops={'alpha': 0.4})
sns.stripplot(x="Conditions", y="β distributions",
            hue="Coefficients", palette=["m","g"],
            data=df_stack, dodge=True, ax=ax)
sns.lineplot(x="Conditions", y="β distributions",
            hue="Coefficients", palette=["m","g"],
            data=df_stack, ci = None, ax=ax)
ax.set_title('Interaction threat risk',
             loc ='left', size = 29)
# Customize ticks
ax.tick_params(bottom=True, left=True, size=5, direction= "in")
ax.tick_params(axis="x", labelsize=20)
ax.tick_params(axis="y", labelsize=20, labelbottom = False)
plt.ylabel("β distributions", fontsize=26)
plt.xlabel("Competition level", fontsize=26)
# Customize legend
handles, labels = ax.get_legend_handles_labels()
l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 0.5), loc=2, 
               borderaxespad=0., fontsize=20, title='Coefficients', 
               title_fontsize=20)

# # Stripplot beta0 with mean line
# fig, ax = plt.subplots(figsize=(6, 6), dpi = 600)
# sns.stripplot(x="Conditions", y="β distributions",
#             hue="Coefficients", palette=["m","g"],
#             data=df_stack, dodge=True, ax=ax)
# sns.lineplot(x="Conditions", y="β distributions",
#             hue="Coefficients", palette=["m","g"],
#             data=df_stack, ci = None, alpha = 0.5, ax=ax)
# ax.set_title('Interaction threat risk',
#               loc ='left', size = 29)
# # Customize ticks
# ax.tick_params(bottom=True, left=True, size=5, direction= "in")
# ax.tick_params(axis="x", labelsize=20)
# ax.tick_params(axis="y", labelsize=20, labelbottom = False)
# plt.ylabel("β distributions", fontsize=26)
# plt.xlabel("Competition level", fontsize=26)
# # Customize legend
# handles, labels = ax.get_legend_handles_labels()
# l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 0.5), loc=2, 
#                 borderaxespad=0., fontsize=20, title='Coefficients', 
#                 title_fontsize=20)
# plt.tick_params(axis='x', which='minor')

# # Stripplot beta1 with mean line
# fig, ax = plt.subplots(figsize=(6, 6), dpi = 600)
# sns.stripplot(x="Conditions", y="β distributions",
#             hue="Coefficients", palette=["g"],
#             data=df_beta1, dodge=True)
# sns.lineplot(x="Conditions", y="β distributions",
#             hue="Coefficients", palette=["g"],
#             data=df_beta1, ci = None, alpha = 0.5)
# ax.set_title('Interaction threat risk',
#               loc ='left', size = 29)
# # Customize ticks
# ax.tick_params(bottom=True, left=True, size=5, direction= "in")
# ax.tick_params(axis="x", labelsize=20)
# ax.tick_params(axis="y", labelsize=20, labelbottom = False)
# plt.ylabel("β distributions", fontsize=26)
# plt.xlabel("Competition level", fontsize=26)
# # Customize legend
# handles, labels = ax.get_legend_handles_labels()
# l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 0.5), loc=2, 
#                 borderaxespad=0., fontsize=20, title='Coefficients', 
#                 title_fontsize=20)
