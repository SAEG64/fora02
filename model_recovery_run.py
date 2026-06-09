#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:33:38 2023
Run and save full model recovery using PyMC

@author: sergej
"""

# %% Data prep functions
""" Prepare Data and Fit Hierarchical Models """

# Requirements
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
import os
import warnings
warnings.filterwarnings('ignore')
import gc
import pickle

path = os.path.dirname(__file__)+"/"
os.chdir(path)

# Select data subset for condition
# 1 for approach forests only
# 2 for avoidance forests only
# If anything else: whole dataset is selected
condition = 0

# List of model names
mdlName = [
    'win stay lose shift',
    '** wait when safe',
    '** binary energy state',
    'weather type',
    '* $\\mathit{r}$ predator',
    '* $\\mathit{p}$ gain',
    'expected gain naive',
    # # '* $\\mathit{p}$ success',
    'marginal value',
    'multi-heuristic policy',
    # '$\mathit{OP}$ values + cap',
    'optimal policy values'
]

# Do NOT standardize for these models
no_standardize_models = [
    '** binary energy state',
    '** wait when safe',
    'weather type',
    'win stay lose shift',
    'marginal value'
]

# Define multi-heuristic policy features
multi_heuristic_features = [
    '* $\\mathit{p}$ gain',
    '* $\\mathit{r}$ predator',
    '** wait when safe',
    '** binary energy state'
]

def compute_bic_from_logp(logp_vals, n_params, n_obs):
    """Compute BIC from log probability values"""
    llf = np.mean(logp_vals)
    return -2 * llf * n_obs + n_params * np.log(n_obs)


def load_and_filter_data(condition):
    """Load all subject data and filter by condition"""
    all_data = []
    all_responses = []
    subject_ids = []
    
    s_idx = 0
    for itr, fle in enumerate(glob.glob(path + "DATA_clean/test_data.*.CAT.csv")):
        sbj = fle[len(path+"DATA_clean/test_data."):-len(".CAT.csv")]
        dt = pd.read_csv(path + "DATA_clean/test_data." + sbj + ".CAT" + ".csv")

        # Add ternary state model
        BNW_state = []
        for index, row in dt.iterrows():
            BNW_state.append(2)
            if row['** binary energy state'] == 1:
                BNW_state[index] = 1
            elif row['** wait when safe'] == 0:
                BNW_state[index] = 3
        dt['ternary state'] = BNW_state
        
        # Filter data
        dt = dt[dt['foraging T/F NaNs'].isnull() == False]
        if condition == 1:
            dt = dt[dt["p/r heuristic"] == "['p']"]
        elif condition == 2:
            dt = dt[dt["p/r heuristic"] == "['r']"]
        dt = dt.reset_index(drop=True)
        
        all_data.append(dt)
        all_responses.append(np.array(dt["fora_response"]))
        subject_ids.append(np.full(len(dt), s_idx))
        s_idx += 1
    
    return all_data, all_responses, subject_ids

# %% MAIN: Fit hierarchical models
""" Main execution: Load data, fit hierarchical models """
# ---------------------------------------------------------------------
# MH INTERACTIONS
# ---------------------------------------------------------------------
# Feature order:
# 0 = p_gain
# 1 = r_predator
# 2 = wait_when_safe
# 3 = binary_energy_state
#
# NOTE: WWS × BES is removed because WWS and BES are mutually exclusive,
# so their product is always zero and makes the design matrix rank-deficient.
interaction_pairs = [
    (0, 1),  # p_gain × r_predator
    (0, 2),  # p_gain × wait_when_safe
    (0, 3),  # p_gain × binary_energy_state
    (1, 2),  # r_predator × wait_when_safe
    (1, 3),  # r_predator × binary_energy_state
]

def compute_interactions(X):
    return np.column_stack([
        X[:, i] * X[:, j]
        for i, j in interaction_pairs
    ])

if __name__ == '__main__':

    # -----------------------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------------------
    all_data, all_responses, subject_ids = load_and_filter_data(condition)
    n_subjects = len(all_data)

    combined_subject_idx = np.concatenate(subject_ids)
    combined_responses = np.concatenate(all_responses)

    fitted_data_per_subject = [pd.DataFrame() for _ in range(n_subjects)]

    # Create directories for storing external idata and compact draws
    os.makedirs(path + "RECOVERY_RESULTS/idata_cache", exist_ok=True)

    idata_list = []
    model_data_info = []

    # -----------------------------------------------------------------
    # FIT EACH MODEL
    # -----------------------------------------------------------------
    for model_idx, nme in enumerate(mdlName):

        print(f"Fitting model {model_idx + 1}/{len(mdlName)}: {nme}")

        # -------------------------------------------------------------
        # PREPARE DESIGN MATRIX
        # -------------------------------------------------------------
        if nme == 'multi-heuristic policy':

            X_list = []

            for feat in multi_heuristic_features:
                feat_data = []
                for subj_data in all_data:
                    if feat in subj_data.columns:
                        feat_data.append(np.array(subj_data[feat]))
                    else:
                        feat_data.append(np.full(len(subj_data), np.nan))
                X_list.append(np.concatenate(feat_data))

            X_full = np.array(X_list).T

            # ---------------------------------------------------------
            # STANDARDIZATION: MH
            # Only continuous features are standardized:
            # 0 = p_gain, 1 = r_predator
            # WWS and BES remain binary.
            # ---------------------------------------------------------
            X_cache = X_full.copy()

            cont_idx = [0, 1]

            X_mean = np.nanmean(X_full[:, cont_idx], axis=0)
            X_std = np.nanstd(X_full[:, cont_idx], axis=0)
            X_std[X_std == 0] = 1.0

            X_cache[:, cont_idx] = (X_full[:, cont_idx] - X_mean) / X_std

            valid_idx = ~(
                np.isnan(X_cache).any(axis=1)
                | np.isnan(combined_responses)
            )

            X = X_cache[valid_idx]
            y = combined_responses[valid_idx]
            subj_idx = combined_subject_idx[valid_idx]
            n_features = X.shape[1]

        else:

            X_list = []

            for subj_data in all_data:
                if nme in subj_data.columns:
                    X_list.append(np.array(subj_data[nme]))
                else:
                    X_list.append(np.full(len(subj_data), np.nan))

            X_full = np.concatenate(X_list)

            # ---------------------------------------------------------
            # STANDARDIZATION: SINGLE-PREDICTOR MODELS
            # ---------------------------------------------------------
            if nme in no_standardize_models:
                X_cache = X_full.copy()
            else:
                X_mean = np.nanmean(X_full)
                X_std = np.nanstd(X_full)

                if X_std == 0:
                    X_std = 1.0

                X_cache = (X_full - X_mean) / X_std

            valid_idx = ~(
                np.isnan(X_cache)
                | np.isnan(combined_responses)
            )

            X = X_cache[valid_idx][:, np.newaxis]
            y = combined_responses[valid_idx]
            subj_idx = combined_subject_idx[valid_idx]
            n_features = 1

        n_obs = len(y)

        # -------------------------------------------------------------
        # BUILD HIERARCHICAL MODEL
        # -------------------------------------------------------------
        with pm.Model() as hmodel:

            # ---------------------------------------------------------
            # RANDOM INTERCEPTS
            # ---------------------------------------------------------
            mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=1)
            sigma_alpha = pm.HalfNormal('sigma_alpha', 1)

            alpha_offset = pm.Normal(
                'alpha_offset',
                mu=0,
                sigma=1,
                shape=n_subjects
            )

            alpha = mu_alpha + sigma_alpha * alpha_offset

            # ---------------------------------------------------------
            # SINGLE-PREDICTOR MODELS
            # ---------------------------------------------------------
            if n_features == 1:

                mu_beta = pm.Normal('mu_beta', mu=0, sigma=1)
                sigma_beta = pm.HalfNormal('sigma_beta', 1)

                beta_offset = pm.Normal(
                    'beta_offset',
                    mu=0,
                    sigma=1,
                    shape=n_subjects
                )

                beta = mu_beta + sigma_beta * beta_offset

                eta = alpha[subj_idx] + beta[subj_idx] * X[:, 0]

            # ---------------------------------------------------------
            # MULTI-HEURISTIC MODEL
            # ---------------------------------------------------------
            else:

                # Main effects: random slopes
                mu_beta_feat = pm.Normal(
                    'mu_beta_feat',
                    mu=0,
                    sigma=1,
                    shape=n_features
                )

                sigma_beta_feat = pm.HalfNormal(
                    'sigma_beta_feat',
                    1,
                    shape=n_features
                )

                beta_offset = pm.Normal(
                    'beta_offset',
                    mu=0,
                    sigma=1,
                    shape=(n_subjects, n_features)
                )

                beta = (
                    mu_beta_feat[np.newaxis, :]
                    + sigma_beta_feat[np.newaxis, :] * beta_offset
                )

                # Pairwise interactions, excluding WWS × BES
                interactions = compute_interactions(X)

                beta_interactions = pm.Normal(
                    'beta_interactions',
                    mu=0,
                    sigma=1,
                    shape=interactions.shape[1]
                )

                eta = (
                    alpha[subj_idx]
                    + pm.math.sum(beta[subj_idx] * X, axis=1)
                    + pm.math.dot(interactions, beta_interactions)
                )

            # ---------------------------------------------------------
            # LIKELIHOOD
            # ---------------------------------------------------------
            y_obs = pm.Bernoulli(
                'y_obs',
                logit_p=eta,
                observed=y
            )

            idata = pm.sample(
                1000,
                tune=1000,
                random_seed=42,
                return_inferencedata=True,
                progressbar=True,
                chains=4,
                cores=4,
                target_accept=0.95
            )

            idata = pm.compute_log_likelihood(idata)

            # POSTERIOR PREDICTIVE: compute mean prediction now (keep only mean)
            with hmodel:
                ppc = pm.sample_posterior_predictive(idata, random_seed=42)

            pred_prob = ppc.posterior_predictive['y_obs'].mean(axis=(0, 1)).values

            # ============================================================
            # SAVE FULL IDATA TO DISK (for archival; not loaded in memory)
            # ============================================================
            safe_model_name = nme.replace(" ", "_").replace("*", "").replace("$", "").replace("\\", "")
            idata_path = path + f"RECOVERY_RESULTS/idata_cache/{model_idx:02d}_{safe_model_name}.pkl"
            try:
                with open(idata_path, 'wb') as f:
                    pickle.dump(idata, f)
                print(f"  Saved full idata to: {idata_path}")
            except Exception as e:
                print(f"  Warning: Could not save idata: {e}")

            # ============================================================
            # EXTRACT COMPACT SET OF POSTERIOR DRAWS FOR SAMPLING
            # ============================================================
            posterior = idata.posterior
            n_chains = posterior.sizes["chain"]
            n_draws = posterior.sizes["draw"]
            n_flat = int(n_chains * n_draws)

            # Keep up to MAX_POSTERIOR_DRAWS per model for synthetic data generation
            keep_k = n_flat
            sel_idx = np.random.choice(n_flat, size=keep_k, replace=False)

            compact_draws = {}
            # always present
            compact_draws["mu_alpha"] = posterior["mu_alpha"].values.reshape(n_flat)[sel_idx]
            compact_draws["sigma_alpha"] = posterior["sigma_alpha"].values.reshape(n_flat)[sel_idx]

            if "mu_beta" in posterior.data_vars:
                compact_draws["mu_beta"] = posterior["mu_beta"].values.reshape(n_flat)[sel_idx]
                compact_draws["sigma_beta"] = posterior["sigma_beta"].values.reshape(n_flat)[sel_idx]
            if "mu_beta_feat" in posterior.data_vars:
                compact_draws["mu_beta_feat"] = posterior["mu_beta_feat"].values.reshape(n_flat, -1)[sel_idx]
                compact_draws["sigma_beta_feat"] = posterior["sigma_beta_feat"].values.reshape(n_flat, -1)[sel_idx]
            if "beta_interactions" in posterior.data_vars:
                compact_draws["beta_interactions"] = posterior["beta_interactions"].values.reshape(n_flat, -1)[sel_idx]

            # store compact draws (not full idata)
            idata_list.append(compact_draws)

            # free large objects (including posterior_predictive and idata)
            try:
                del posterior
                del idata
                del ppc
            except Exception:
                pass
            gc.collect()

        # (posterior predictive was computed above and stored in `pred_prob`)

        # -------------------------------------------------------------
        # MAP PREDICTIONS BACK TO SUBJECT FILES
        # -------------------------------------------------------------
        for subj_id in range(n_subjects):

            n_obs_subj = len(all_data[subj_id])
            subj_mask = (subj_idx == subj_id)
            subj_pred = np.full(n_obs_subj, np.nan)

            if nme == 'multi-heuristic policy':

                orig_valid_idx = ~(
                    pd.isna(all_data[subj_id][multi_heuristic_features[0]]).values
                )

                for feat in multi_heuristic_features[1:]:
                    orig_valid_idx &= ~(
                        pd.isna(all_data[subj_id][feat]).values
                    )

            else:

                if nme in all_data[subj_id].columns:
                    orig_valid_idx = ~(
                        pd.isna(all_data[subj_id][nme]).values
                    )
                else:
                    orig_valid_idx = np.full(n_obs_subj, False)

            orig_valid_idx &= ~(
                pd.isna(all_data[subj_id]['fora_response']).values
            )

            subj_pred[orig_valid_idx] = pred_prob[subj_mask]

            fitted_data_per_subject[subj_id][nme + ' fit'] = subj_pred

        # -------------------------------------------------------------
        # Note: compact draws already stored into `idata_list` above.
        # -------------------------------------------------------------
        model_data_info.append({
            'model_name': nme,
            'n_obs': n_obs,
            'n_subjects': n_subjects,
            'n_features': n_features,
            'n_interactions': len(interaction_pairs) if nme == 'multi-heuristic policy' else 0,
        })

    # -----------------------------------------------------------------
    # SAVE FITTED DATA
    # -----------------------------------------------------------------
    for subj_id, fle in enumerate(glob.glob(path + "DATA_clean/test_data.*.CAT.csv")):

        sbj = fle[len(path + "DATA_clean/test_data."):-len(".CAT.csv")]

        dt_with_fits = pd.concat(
            [all_data[subj_id], fitted_data_per_subject[subj_id]],
            axis=1
        )

        dt_with_fits.to_csv(
            path
            + "DATA_clean/DATA_fitted/test_data."
            + sbj
            + ".CAT"
            + "_regress"
            + ".csv",
            index=False
        )

# %% Model + parameter recovery
""" Model + parameter recovery """

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

print("\n" + "="*80)
print("MODEL + PARAMETER RECOVERY (FULL: MODEL + PARAMETER)")
print("="*80)

# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------
n_sim = 100
n_models = len(mdlName)
n_jobs = min(2, max(1, multiprocessing.cpu_count() /2))
lapse_rate = 0.0

# ---------------------------------------------------------------------
# SAFE NAMING
# ---------------------------------------------------------------------
def make_safe(name):
    return (
        name.replace(" ", "_")
            .replace("*", "")
            .replace("$", "")
            .replace("\\", "")
            .replace("{", "")
            .replace("}", "")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
    )

feature_names = [make_safe(f) for f in multi_heuristic_features]

interaction_names = [
    f"{feature_names[i]}_x_{feature_names[j]}"
    for i, j in interaction_pairs
]

# ---------------------------------------------------------------------
# DESIGN MATRIX
# ---------------------------------------------------------------------
def build_design_matrix(nme):

    if nme == 'multi-heuristic policy':
        X_list = []
        for feat in multi_heuristic_features:
            feat_data = []
            for subj_data in all_data:
                if feat in subj_data.columns:
                    feat_data.append(np.array(subj_data[feat]))
                else:
                    feat_data.append(np.full(len(subj_data), np.nan))
            X_list.append(np.concatenate(feat_data))
        X_full = np.array(X_list).T

    else:
        X_list = []
        for subj_data in all_data:
            if nme in subj_data.columns:
                X_list.append(np.array(subj_data[nme]))
            else:
                X_list.append(np.full(len(subj_data), np.nan))
        X_full = np.concatenate(X_list)[:, None]

    return X_full

# ---------------------------------------------------------------------
# GLOBAL MASK
# ---------------------------------------------------------------------
def build_global_mask():

    masks = []

    for nme in mdlName:
        X_full = build_design_matrix(nme)
        mask = ~(np.isnan(X_full).any(axis=1) | np.isnan(combined_responses))
        masks.append(mask)

    return np.logical_and.reduce(masks)

# ---------------------------------------------------------------------
# MH INTERACTIONS
# ---------------------------------------------------------------------
# Feature order:
# 0 = p_gain
# 1 = r_predator
# 2 = wait_when_safe
# 3 = binary_energy_state
#
# NOTE: WWS × BES is removed because WWS and BES are mutually exclusive,
# so their product is always zero and makes the design matrix rank-deficient.
interaction_pairs = [
    (0, 1),  # p_gain × r_predator
    (0, 2),  # p_gain × wait_when_safe
    (0, 3),  # p_gain × binary_energy_state
    (1, 2),  # r_predator × wait_when_safe
    (1, 3),  # r_predator × binary_energy_state
]

def compute_interactions(X):
    return np.column_stack([
        X[:, i] * X[:, j]
        for i, j in interaction_pairs
    ])

# ---------------------------------------------------------------------
# POSTERIOR DRAW
# ---------------------------------------------------------------------
def sample_group_draw(idata, n_features):

    # Support compact draw dicts produced during the initial fit phase
    if isinstance(idata, dict):
        draws_k = len(idata["mu_alpha"])
        idx = np.random.randint(draws_k)

        mu_alpha = float(idata["mu_alpha"][idx])
        sigma_alpha = float(idata["sigma_alpha"][idx])

        if n_features == 1:
            mu_beta = np.array([float(idata["mu_beta"][idx])], dtype=float)
            sigma_beta = np.array([float(idata["sigma_beta"][idx])], dtype=float)
        else:
            mu_beta = idata["mu_beta_feat"][idx].astype(float)
            sigma_beta = idata["sigma_beta_feat"][idx].astype(float)

        beta_int = None
        if "beta_interactions" in idata:
            beta_int = idata["beta_interactions"][idx].astype(float)

        return mu_alpha, sigma_alpha, mu_beta, sigma_beta, beta_int

    # Fallback: accept full InferenceData (older behavior)
    posterior = idata.posterior

    c = np.random.randint(posterior.sizes["chain"])
    d = np.random.randint(posterior.sizes["draw"])

    mu_alpha = float(posterior["mu_alpha"].values[c, d])
    sigma_alpha = float(posterior["sigma_alpha"].values[c, d])

    if n_features == 1:
        mu_beta = np.array([posterior["mu_beta"].values[c, d]], dtype=float)
        sigma_beta = np.array([posterior["sigma_beta"].values[c, d]], dtype=float)
    else:
        mu_beta = posterior["mu_beta_feat"].values[c, d].astype(float)
        sigma_beta = posterior["sigma_beta_feat"].values[c, d].astype(float)

    beta_int = None
    if "beta_interactions" in posterior.data_vars:
        beta_int = posterior["beta_interactions"].values[c, d].astype(float)

    return mu_alpha, sigma_alpha, mu_beta, sigma_beta, beta_int

# ---------------------------------------------------------------------
# SIMULATION
# ---------------------------------------------------------------------
def simulate_dataset(
    X,
    subj_idx,
    mu_alpha,
    sigma_alpha,
    mu_beta,
    sigma_beta,
    beta_int,
    lapse_rate=lapse_rate
):

    n_subjects_local = int(np.max(subj_idx)) + 1

    # Explicit non-centered generative draw
    alpha_offset_true = np.random.normal(0, 1, n_subjects_local)
    alpha_true = mu_alpha + sigma_alpha * alpha_offset_true

    beta_offset_true = np.random.normal(
        0,
        1,
        size=(n_subjects_local, len(mu_beta))
    )
    beta_true = mu_beta[np.newaxis, :] + sigma_beta[np.newaxis, :] * beta_offset_true

    eta = alpha_true[subj_idx] + np.sum(beta_true[subj_idx] * X, axis=1)

    if beta_int is not None:
        inter = compute_interactions(X)
        if inter is not None:
            eta += inter @ beta_int

    p = 1 / (1 + np.exp(-eta))
    y = np.random.binomial(1, p)

    if lapse_rate > 0:
        lapse_mask = np.random.binomial(1, lapse_rate, size=len(y)).astype(bool)
        y[lapse_mask] = np.random.binomial(1, 0.5, size=np.sum(lapse_mask))

    return {
        "y": y,
        "alpha_offset_true": alpha_offset_true,
        "alpha_true": alpha_true,
        "beta_offset_true": beta_offset_true,
        "beta_true": beta_true,
    }

# ---------------------------------------------------------------------
# MODEL FIT
# ---------------------------------------------------------------------
def fit_model(X, y, subj_idx, fit_name):

    try:
        with pm.Model() as model:

            mu_alpha = pm.Normal("mu_alpha", 0, 1)
            sigma_alpha = pm.HalfNormal("sigma_alpha", 1)

            alpha_offset = pm.Normal("alpha_offset", 0, 1, shape=n_subjects)
            alpha = mu_alpha + sigma_alpha * alpha_offset

            if fit_name == "multi-heuristic policy":

                n_feat = X.shape[1]

                mu_beta = pm.Normal("mu_beta_feat", 0, 1, shape=n_feat)
                sigma_beta = pm.HalfNormal("sigma_beta_feat", 1, shape=n_feat)

                beta_offset = pm.Normal(
                    "beta_offset",
                    0,
                    1,
                    shape=(n_subjects, n_feat)
                )
                beta = mu_beta[np.newaxis, :] + sigma_beta[np.newaxis, :] * beta_offset

                eta_main = pm.math.sum(beta[subj_idx] * X, axis=1)

                inter = compute_interactions(X)

                if inter is not None:
                    beta_int = pm.Normal(
                        "beta_interactions",
                        0,
                        1,
                        shape=inter.shape[1]
                    )
                    eta_inter = pm.math.dot(inter, beta_int)
                    eta = alpha[subj_idx] + eta_main + eta_inter
                else:
                    eta = alpha[subj_idx] + eta_main

            else:
                mu_beta = pm.Normal("mu_beta", 0, 1)
                sigma_beta = pm.HalfNormal("sigma_beta", 1)

                beta_offset = pm.Normal("beta_offset", 0, 1, shape=n_subjects)
                beta = mu_beta + sigma_beta * beta_offset

                eta = alpha[subj_idx] + beta[subj_idx] * X[:, 0]

            pm.Bernoulli("y_obs", logit_p=eta, observed=y)

            idata = pm.sample(
                draws=500,
                tune=500,
                chains=4,
                cores=1,
                progressbar=False,
                target_accept=0.95,
                return_inferencedata=True
            )

            idata = pm.compute_log_likelihood(idata)

        # Extract posterior means and standard errors over sampling dims only.
        # This preserves subject-level dimensions (e.g., alpha_offset, beta_offset).
        posterior = idata.posterior
        reduce_dims = [d for d in ("chain", "draw") if d in posterior.dims]
        posterior_mean = posterior.mean(dim=reduce_dims)
        posterior_std = posterior.std(dim=reduce_dims)

        # Compute ELPD via LOO for model comparison
        try:
            loo_elpd = az.loo(idata).elpd_loo
        except Exception:
            loo_elpd = -np.inf

        # Diagnostics
        diag = get_fit_diagnostics(idata)

        # Return compact summary (not full idata)
        fit_result = {
            "posterior_mean": posterior_mean,
            "posterior_std": posterior_std,
            "elpd_loo": loo_elpd,
            "rhat_max": diag["rhat_max"],
            "ess_min": diag["ess_min"],
        }

        # Free memory
        try:
            del posterior
            del idata
        except Exception:
            pass
        gc.collect()

        return fit_result

    except Exception as e:
        print(f"Model failed: {fit_name} | Error: {e}")
        return None

# ---------------------------------------------------------------------
# DIAGNOSTICS
# ---------------------------------------------------------------------
def get_fit_diagnostics(idata):

    var_names = ["mu_alpha", "sigma_alpha"]

    post_vars = list(idata.posterior.data_vars)

    if "mu_beta" in post_vars:
        var_names += ["mu_beta", "sigma_beta"]

    if "mu_beta_feat" in post_vars:
        var_names += ["mu_beta_feat", "sigma_beta_feat"]

    if "beta_interactions" in post_vars:
        var_names += ["beta_interactions"]

    try:
        rhat = az.rhat(idata, var_names=var_names).to_array().values
        ess = az.ess(idata, var_names=var_names).to_array().values

        return {
            "rhat_max": float(np.nanmax(rhat)),
            "ess_min": float(np.nanmin(ess)),
        }

    except Exception as e:
        print(f"Diagnostics failed: {e}")
        return {
            "rhat_max": np.nan,
            "ess_min": np.nan,
        }

# ---------------------------------------------------------------------
# RECONSTRUCT POSTERIOR SUBJECT PARAMETERS
# ---------------------------------------------------------------------
def reconstruct_subject_parameters(post):

    alpha_draws = (
        post["mu_alpha"].values[:, :, None] +
        post["sigma_alpha"].values[:, :, None] *
        post["alpha_offset"].values
    )
    alpha_est = alpha_draws.mean(axis=(0, 1))

    if "mu_beta" in post.data_vars:
        beta_draws = (
            post["mu_beta"].values[:, :, None] +
            post["sigma_beta"].values[:, :, None] *
            post["beta_offset"].values
        )
        beta_est = beta_draws.mean(axis=(0, 1))[:, None]

    elif "mu_beta_feat" in post.data_vars:
        beta_draws = (
            post["mu_beta_feat"].values[:, :, None, :] +
            post["sigma_beta_feat"].values[:, :, None, :] *
            post["beta_offset"].values
        )
        beta_est = beta_draws.mean(axis=(0, 1))

    else:
        beta_est = None

    return alpha_est, beta_est

# ---------------------------------------------------------------------
# PRECOMPUTE DATA
# ---------------------------------------------------------------------
global_mask = build_global_mask()

X_cache = {}
for name in mdlName:

    X_full = build_design_matrix(name)
    X_full = X_full[global_mask]

    if name == "multi-heuristic policy":

        X_tmp = X_full.copy()
        cont_idx = [0, 1]

        X_mean = np.nanmean(X_full[:, cont_idx], axis=0)
        X_std = np.nanstd(X_full[:, cont_idx], axis=0)
        X_std[X_std == 0] = 1.0

        X_tmp[:, cont_idx] = (X_full[:, cont_idx] - X_mean) / X_std
        X_cache[name] = X_tmp

    elif name in no_standardize_models:
        X_cache[name] = X_full.copy()

    else:
        X_mean = np.nanmean(X_full, axis=0)
        X_std = np.nanstd(X_full, axis=0)
        X_std[X_std == 0] = 1.0

        X_cache[name] = (X_full - X_mean) / X_std

subj_idx_base = combined_subject_idx[global_mask]

# =====================================================================
# EXTRACT SUBJECT-LEVEL PARAMETER ESTIMATES (from posterior means)
# =====================================================================
def extract_alpha_estimates(post_mean, n_subjects):
    """Extract subject-level alpha estimates from posterior mean."""
    mu_alpha = float(post_mean["mu_alpha"].values)
    sigma_alpha = float(post_mean["sigma_alpha"].values) if "sigma_alpha" in post_mean else 1.0
    alpha_offset = post_mean["alpha_offset"].values if "alpha_offset" in post_mean else np.zeros(n_subjects)
    return mu_alpha + sigma_alpha * alpha_offset

def extract_beta_estimates(post_mean, fit_name, n_subjects):
    """Extract subject-level beta estimates from posterior mean."""
    if fit_name == "multi-heuristic policy":
        if "mu_beta_feat" in post_mean:
            mu_beta = post_mean["mu_beta_feat"].values
            sigma_beta = post_mean["sigma_beta_feat"].values if "sigma_beta_feat" in post_mean else np.ones_like(mu_beta)
            beta_offset = post_mean["beta_offset"].values if "beta_offset" in post_mean else np.zeros((n_subjects, len(mu_beta)))
            return mu_beta[np.newaxis, :] + sigma_beta[np.newaxis, :] * beta_offset
        else:
            return np.zeros((n_subjects, 1))
    else:
        if "mu_beta" in post_mean:
            mu_beta = float(post_mean["mu_beta"].values)
            sigma_beta = float(post_mean["sigma_beta"].values) if "sigma_beta" in post_mean else 1.0
            beta_offset = post_mean["beta_offset"].values if "beta_offset" in post_mean else np.zeros(n_subjects)
            return (mu_beta + sigma_beta * beta_offset).reshape(-1, 1)
        else:
            return np.zeros((n_subjects, 1))

# =====================================================================
# SINGLE SIMULATION
# =====================================================================
def run_single_sim(gen_idx, sim_idx):

    np.random.seed(gen_idx * 1000 + sim_idx)

    idata_gen = idata_list[gen_idx]
    gen_name = mdlName[gen_idx]
    n_features = model_data_info[gen_idx]["n_features"]

    print(f"\n[SIM {sim_idx}] Generating data from: {gen_name}")

    X_real = X_cache[gen_name]

    mu_alpha, sigma_alpha, mu_beta, sigma_beta, beta_int = sample_group_draw(
        idata_gen,
        n_features
    )

    sim = simulate_dataset(
        X=X_real,
        subj_idx=subj_idx_base,
        mu_alpha=mu_alpha,
        sigma_alpha=sigma_alpha,
        mu_beta=mu_beta,
        sigma_beta=sigma_beta,
        beta_int=beta_int,
        lapse_rate=lapse_rate
    )

    y_sim = sim["y"]
    alpha_true = sim["alpha_true"]
    beta_true = sim["beta_true"]

    loo_scores = []
    param_result = None
    all_fit_diagnostics = []

    for fit_idx, fit_name in enumerate(mdlName):

        print(f"  → Fitting model: {fit_name}")

        X_fit = X_cache[fit_name]
        fit_result = fit_model(X_fit, y_sim, subj_idx_base, fit_name)

        if fit_result is None:
            print(f"Fit failed: gen={gen_name}, fit={fit_name}, sim={sim_idx}")
            loo_val = -np.inf
            fit_diag = {
                "gen_model": gen_name,
                "fit_model": fit_name,
                "sim_idx": sim_idx,
                "gen_idx": gen_idx,
                "loo_elpd": loo_val,
                "rhat_max": np.nan,
                "ess_min": np.nan,
            }

        else:
            loo_val = fit_result["elpd_loo"]
            rhat_max = fit_result["rhat_max"]
            ess_min = fit_result["ess_min"]

            fit_diag = {
                "gen_model": gen_name,
                "fit_model": fit_name,
                "sim_idx": sim_idx,
                "gen_idx": gen_idx,
                "loo_elpd": loo_val,
                "rhat_max": rhat_max,
                "ess_min": ess_min,
            }

            if fit_name == gen_name:
                # Extract estimated parameters from posterior means
                post_mean = fit_result["posterior_mean"]
                post_std = fit_result["posterior_std"]

                # Reconstruct subject-level parameters from means
                alpha_est = extract_alpha_estimates(post_mean, n_subjects)
                beta_est = extract_beta_estimates(post_mean, fit_name, n_subjects)

                param_result = {
                    "gen_model": gen_name,
                    "fit_model": fit_name,
                    "sim_idx": sim_idx,
                    "gen_idx": gen_idx,

                    # group-level intercept distribution
                    "true_mu_alpha": float(mu_alpha),
                    "est_mu_alpha": float(post_mean["mu_alpha"].values),
                    "true_sigma_alpha": float(sigma_alpha),
                    "est_sigma_alpha": float(post_mean["sigma_alpha"].values),

                    # reconstructed subject-level intercept distribution
                    "true_alpha_subject_mean": float(np.mean(alpha_true)),
                    "est_alpha_subject_mean": float(np.mean(alpha_est)),
                    "true_alpha_subject_sd": float(np.std(alpha_true, ddof=1)),
                    "est_alpha_subject_sd": float(np.std(alpha_est, ddof=1)),

                    "rhat_max": rhat_max,
                    "ess_min": ess_min,
                }

                # per-subject intercepts
                for s in range(len(alpha_true)):
                    param_result[f"true_alpha_subj_{s:02d}"] = float(alpha_true[s])
                    param_result[f"est_alpha_subj_{s:02d}"] = float(alpha_est[s])

                # single-feature model
                if "mu_beta" in post_mean.data_vars:

                    param_result["true_beta"] = float(mu_beta[0])
                    param_result["est_beta"] = float(post_mean["mu_beta"].values)

                    param_result["true_sigma_beta"] = float(sigma_beta[0])
                    param_result["est_sigma_beta"] = float(post_mean["sigma_beta"].values)

                    param_result["true_beta_subject_mean"] = float(np.mean(beta_true[:, 0]))
                    param_result["est_beta_subject_mean"] = float(np.mean(beta_est[:, 0]))
                    param_result["true_beta_subject_sd"] = float(np.std(beta_true[:, 0], ddof=1))
                    param_result["est_beta_subject_sd"] = float(np.std(beta_est[:, 0], ddof=1))

                    for s in range(beta_true.shape[0]):
                        param_result[f"true_beta_subj_{s:02d}"] = float(beta_true[s, 0])
                        param_result[f"est_beta_subj_{s:02d}"] = float(beta_est[s, 0])

                # multi-feature model
                elif "mu_beta_feat" in post_mean.data_vars:

                    est_mu_beta = post_mean["mu_beta_feat"].values
                    est_sigma_beta = post_mean["sigma_beta_feat"].values

                    for i, fname in enumerate(feature_names):

                        # group-level slope distribution
                        param_result[f"true_beta_{fname}"] = float(mu_beta[i])
                        param_result[f"est_beta_{fname}"] = float(est_mu_beta[i])

                        param_result[f"true_sigma_beta_{fname}"] = float(sigma_beta[i])
                        param_result[f"est_sigma_beta_{fname}"] = float(est_sigma_beta[i])

                        # reconstructed subject-level slope distribution
                        param_result[f"true_beta_subject_mean_{fname}"] = float(np.mean(beta_true[:, i]))
                        param_result[f"est_beta_subject_mean_{fname}"] = float(np.mean(beta_est[:, i]))
                        param_result[f"true_beta_subject_sd_{fname}"] = float(np.std(beta_true[:, i], ddof=1))
                        param_result[f"est_beta_subject_sd_{fname}"] = float(np.std(beta_est[:, i], ddof=1))

                        # per-subject slopes
                        for s in range(beta_true.shape[0]):
                            param_result[f"true_beta_{fname}_subj_{s:02d}"] = float(beta_true[s, i])
                            param_result[f"est_beta_{fname}_subj_{s:02d}"] = float(beta_est[s, i])

                    # fixed interactions
                    if "beta_interactions" in post_mean.data_vars and beta_int is not None:

                        est_beta_int = post_mean["beta_interactions"].values

                        for i, iname in enumerate(interaction_names):
                            param_result[f"true_beta_int_{iname}"] = float(beta_int[i])
                            param_result[f"est_beta_int_{iname}"] = float(est_beta_int[i])

        all_fit_diagnostics.append(fit_diag)
        loo_scores.append(loo_val)

    best_model = int(np.argmax(loo_scores))

    return gen_idx, best_model, param_result, all_fit_diagnostics

# ---------------------------------------------------------------------
# RUN FULL RECOVERY
# ---------------------------------------------------------------------
results = Parallel(n_jobs=n_jobs)(
    delayed(run_single_sim)(g, s)
    for g in range(n_models)
    for s in range(n_sim)
)

# ---------------------------------------------------------------------
# COLLECT RESULTS
# ---------------------------------------------------------------------
recovery_counts = np.zeros((n_models, n_models))
param_recovery = []
all_diagnostics = []

for gen_idx, best_model, param, fit_diags in results:
    recovery_counts[gen_idx, best_model] += 1
    if param is not None:
        param_recovery.append(param)
    all_diagnostics.extend(fit_diags)

recovery_matrix = recovery_counts / n_sim

# ---------------------------------------------------------------------
# SAVE EVERYTHING
# ---------------------------------------------------------------------
os.makedirs(path + "RECOVERY_RESULTS", exist_ok=True)

np.save(path + "RECOVERY_RESULTS/model_names.npy", np.array(mdlName))
np.save(path + "RECOVERY_RESULTS/recovery_counts.npy", recovery_counts)
np.save(path + "RECOVERY_RESULTS/recovery_matrix.npy", recovery_matrix)

df_param = pd.DataFrame(param_recovery)
df_param.to_csv(path + "RECOVERY_RESULTS/param_recovery.csv", index=False)

df_diags = pd.DataFrame(all_diagnostics)
df_diags.to_csv(path + "RECOVERY_RESULTS/all_fit_diagnostics.csv", index=False)

print("\nSaved model + parameter recovery outputs.")