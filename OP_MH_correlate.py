# %% MH PREDICTS OPTIMAL POLICY WITH CONDITION INTERACTION
"""Test whether MH approximates OP better depending on condition"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import os

path = os.path.dirname(__file__) + "/"

# ==============================================================================
# 1. DATA PREPARATION
# ==============================================================================

df = pd.read_csv(path + 'DATA_clean/DATA_group_level/datall_with_condition_order.csv')

# ------------------------------------------------------------------------------
# SUBJECT INDEX
# ------------------------------------------------------------------------------
df["subj_idx"], subj_ids = pd.factorize(df["Subject_ID"])
n_subj = len(subj_ids)
subj_idx = df["subj_idx"].values.astype(int)

# ------------------------------------------------------------------------------
# VARIABLES
# ------------------------------------------------------------------------------
op_cap = df["OP_value_difference"].values.astype(float)
mh = df["multi-heuristic policy"].values.astype(float)

# Standardize predictors (important!)
op_cap = (op_cap - op_cap.mean()) / op_cap.std()
mh = (mh - mh.mean()) / mh.std()

# ------------------------------------------------------------------------------
# CONDITION CODING (same as mixture model)
# ------------------------------------------------------------------------------
cond = 1 - pd.factorize(df["condition_rORp"])[0]
cond_signed = 2 * cond - 1   # -1 / +1

# ==============================================================================
# 2. MODEL
# ==============================================================================

with pm.Model() as regression_model:

    # --------------------------------------------------------------------------
    # GROUP-LEVEL PRIORS
    # --------------------------------------------------------------------------
    mu_beta0 = pm.Normal("mu_beta0", 0, 1.0)
    sigma_beta0 = pm.HalfNormal("sigma_beta0", 1.0)

    mu_beta_mh = pm.Normal("mu_beta_mh", 0, 1.0)
    sigma_beta_mh = pm.HalfNormal("sigma_beta_mh", 1.0)

    beta_cond = pm.Normal("beta_cond", 0, 1.0)
    beta_interact = pm.Normal("beta_interact", 0, 1.0)

    # --------------------------------------------------------------------------
    # SUBJECT-LEVEL OFFSETS
    # --------------------------------------------------------------------------
    beta0_offset = pm.Normal("beta0_offset", 0, 1, shape=n_subj)
    beta_mh_offset = pm.Normal("beta_mh_offset", 0, 1, shape=n_subj)

    # --------------------------------------------------------------------------
    # SUBJECT-LEVEL PARAMETERS (TRACKED!)
    # --------------------------------------------------------------------------
    beta0 = pm.Deterministic(
        "beta0",
        mu_beta0 + beta0_offset * sigma_beta0
    )

    beta_mh = pm.Deterministic(
        "beta_mh",
        mu_beta_mh + beta_mh_offset * sigma_beta_mh
    )

    # --------------------------------------------------------------------------
    # LINEAR MODEL
    # --------------------------------------------------------------------------
    mu = (
        beta0[subj_idx]
        + beta_mh[subj_idx] * mh
        + beta_cond * cond_signed
        + beta_interact * mh * cond_signed
    )

    # --------------------------------------------------------------------------
    # NOISE
    # --------------------------------------------------------------------------
    sigma = pm.HalfNormal("sigma", 1.0)

    # --------------------------------------------------------------------------
    # LIKELIHOOD
    # --------------------------------------------------------------------------
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=op_cap)

    # ==============================================================================
    # 3. SAMPLING
    # ==============================================================================
    idata_reg = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=0.95,
        random_seed=42
    )

# ==============================================================================
# 4. ANALYSIS
# ==============================================================================

print("\n=== MODEL SUMMARY ===")
print(az.summary(idata_reg, var_names=[
    "mu_beta_mh",
    "beta_cond",
    "beta_interact"
]))

# ------------------------------------------------------------------------------
# INTERACTION EFFECT
# ------------------------------------------------------------------------------
beta_interact_samples = idata_reg.posterior["beta_interact"].values.flatten()

print("\n=== INTERACTION EFFECT ===")
print("Mean:", beta_interact_samples.mean())
print("P(MH better predicts OP in avoidance):",
      (beta_interact_samples > 0).mean())

# ------------------------------------------------------------------------------
# SUBJECT-LEVEL MH → OP SLOPES
# ------------------------------------------------------------------------------
beta_mh_subj = idata_reg.posterior["beta_mh"].mean(dim=("chain", "draw")).values

df_subj = pd.DataFrame({
    "subject": subj_ids,
    "mh_slope": beta_mh_subj
})

print("\n=== SUBJECT-LEVEL MH → OP SLOPES ===")
print(df_subj.head())

# ------------------------------------------------------------------------------
# OPTIONAL: CONDITION-SPECIFIC SLOPES
# ------------------------------------------------------------------------------
# slope in avoidance (+1) vs approach (-1)
beta_interact_mean = beta_interact_samples.mean()

slope_approach = beta_mh_subj - beta_interact_mean
slope_avoidance = beta_mh_subj + beta_interact_mean

df_subj["mh_slope_approach"] = slope_approach
df_subj["mh_slope_avoidance"] = slope_avoidance

print("\n=== CONDITION-SPECIFIC SLOPES ===")
print(df_subj.head())