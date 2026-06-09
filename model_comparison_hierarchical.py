#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:33:38 2023
Hierarchical Bayesian analysis using PyMC

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
    # '* $\\mathit{p}$ success',
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
        
        # Build hierarchical PyMC model
        with pm.Model() as hmodel:
            # Group-level hyperpriors for intercept
            mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=1)
            sigma_alpha = pm.Exponential('sigma_alpha', 1)
            
            # Subject-level intercept parameters (non-centered parameterization)
            alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=n_subjects)
            alpha = mu_alpha + sigma_alpha * alpha_offset
            
            if n_features == 1:
                # Single feature model
                mu_beta = pm.Normal('mu_beta', mu=0, sigma=1)
                sigma_beta = pm.Exponential('sigma_beta', 1)
                beta_offset = pm.Normal('beta_offset', mu=0, sigma=1, shape=n_subjects)
                beta = mu_beta + sigma_beta * beta_offset
                # Linear predictor
                eta = alpha[subj_idx] + beta[subj_idx] * X[:, 0]
            else:
                # Multi-feature model: RANDOM SLOPES for main effects + FIXED interactions
                # =====================================================================
                # Main effects: p_gain, predator, wait_when_safe, binary_energy_state
                # with group-level hyperpriors and subject-level random slopes
                mu_beta_feat = pm.Normal('mu_beta_feat', mu=0, sigma=1, shape=n_features)
                sigma_beta_feat = pm.Exponential('sigma_beta_feat', 1, shape=n_features)
                
                # Subject deviations for random slopes (non-centered parameterization)
                beta_offset = pm.Normal('beta_offset', mu=0, sigma=1, shape=(n_subjects, n_features))
                beta = mu_beta_feat[np.newaxis, :] + sigma_beta_feat[np.newaxis, :] * beta_offset
                
                # Interaction terms (pairwise products) - FIXED GROUP-LEVEL ONLY
                # =====================================================================
                # This captures synergistic effects without individual variation
                # NOTE: WWS × BES is excluded (mutually exclusive variables)
                interactions = np.array([
                    X[:, i] * X[:, j]
                    for i, j in interaction_pairs
                ]).T  # Shape: (n_obs, n_interactions)
                
                n_interactions = interactions.shape[1]
                
                # Group-level FIXED coefficients for pairwise interactions (no subject variation)
                beta_interactions = pm.Normal('beta_interactions', mu=0, sigma=1, shape=n_interactions)
                
                # Linear predictor for multi-feature model with interactions
                eta = (alpha[subj_idx] + 
                       pm.math.sum(beta[subj_idx] * X, axis=1) +
                       pm.math.sum(beta_interactions * interactions, axis=1))
            
            # Likelihood
            y_obs = pm.Bernoulli('y_obs', logit_p=eta, observed=y)
            
            # Sample from posterior with improved settings for convergence
            # Increased tuning and draws; target_accept improves sampler efficiency
            idata = pm.sample(1000, tune=1000, random_seed=42,
                            return_inferencedata=True,
                            progressbar=True, chains=4, cores=4,
                            target_accept=0.9)
            
            # Compute log_likelihood for LOO calculation (assign return value)
            idata = pm.compute_log_likelihood(idata)
        
        
        # Compute posterior predictive mean for each observation
        with hmodel:
            ppc = pm.sample_posterior_predictive(idata, random_seed=42)
        
        pred_prob = ppc.posterior_predictive['y_obs'].mean(axis=(0, 1)).values
        
        # Store fitted predictions per subject
        obs_idx = 0
        for subj_id in range(n_subjects):
            n_obs_subj = len(all_data[subj_id])
            subj_mask = (subj_idx == subj_id)
            subj_pred = np.full(n_obs_subj, np.nan)
            
            # Map predictions back to original subject data
            orig_valid_idx = np.full(n_obs_subj, False)
            
            if nme == 'multi-heuristic policy':
                orig_valid_idx = ~(pd.isna(all_data[subj_id][multi_heuristic_features[0]]).values)
                for feat in multi_heuristic_features[1:]:
                    orig_valid_idx &= ~(pd.isna(all_data[subj_id][feat]).values)
            else:
                if nme in all_data[subj_id].columns:
                    orig_valid_idx = ~(pd.isna(all_data[subj_id][nme]).values)
            
            orig_valid_idx &= ~(pd.isna(all_data[subj_id]['fora_response']).values)
            subj_pred[orig_valid_idx] = pred_prob[subj_mask]
            
            fitted_data_per_subject[subj_id][nme + ' fit'] = subj_pred
        
        # Store idata and model info for LOO comparison
        idata_list.append(idata)
        model_data_info.append({
            'model_name': nme,
            'n_obs': n_obs,
            'n_subjects': n_subjects,
            'n_features': n_features
        })
    
    # Save fitted data per subject
    for subj_id, fle in enumerate(glob.glob(path + "DATA_clean/test_data.*.CAT.csv")):
        sbj = fle[len(path+"DATA_clean/test_data."):-len(".CAT.csv")]
        dt_with_fits = pd.concat([all_data[subj_id], fitted_data_per_subject[subj_id]], axis=1)
        dt_with_fits.to_csv(path + "DATA_clean/DATA_fitted/test_data." + sbj + ".CAT" + "_regress" + ".csv", index=False)

# %% Diagnostics: Fit Two Winning Models with PPC for Publication-Ready Plots
""" Fit Multi-Heuristic and Optimal Policy Models with Posterior Predictive Checks """
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Fitting Two Winning Models with Posterior Predictive Checks")
    print("="*60)
    
    # Hardcoded winning models
    winning_models = ['multi-heuristic policy', 'optimal policy values']
    model_display_names = ['Multi-Feature Policy', r'$\Delta Q$-Values']
    
    # Reload data for winning models
    all_data_ppc, all_responses_ppc, subject_ids_ppc = load_and_filter_data(condition)
    n_subjects_ppc = len(all_data_ppc)
    
    combined_subject_idx_ppc = np.concatenate(subject_ids_ppc)
    combined_responses_ppc = np.concatenate(all_responses_ppc)
    
    ppc_results = []  # Store ppc samples and observed data
    
    for mdl_idx, (nme, display_name) in enumerate(zip(winning_models, model_display_names)):
        print(f"\nFitting: {nme}")
        
        # --------- PREPARE DESIGN MATRIX ---------
        if nme == 'multi-heuristic policy':
            X_list = []
            for feat in multi_heuristic_features:
                feat_data = []
                for subj_data in all_data_ppc:
                    if feat in subj_data.columns:
                        feat_data.append(np.array(subj_data[feat]))
                    else:
                        feat_data.append(np.full(len(subj_data), np.nan))
                X_list.append(np.concatenate(feat_data))
            
            X_full = np.array(X_list).T
            
            # Standardization: MH
            X_cache = X_full.copy()
            cont_idx = [0, 1]
            X_mean = np.nanmean(X_full[:, cont_idx], axis=0)
            X_std = np.nanstd(X_full[:, cont_idx], axis=0)
            X_std[X_std == 0] = 1.0
            X_cache[:, cont_idx] = (X_full[:, cont_idx] - X_mean) / X_std
            
            valid_idx = ~(np.isnan(X_cache).any(axis=1) | np.isnan(combined_responses_ppc))
            X = X_cache[valid_idx]
            y = combined_responses_ppc[valid_idx]
            subj_idx = combined_subject_idx_ppc[valid_idx]
            n_features = X.shape[1]
        else:
            X_list = []
            for subj_data in all_data_ppc:
                if nme in subj_data.columns:
                    X_list.append(np.array(subj_data[nme]))
                else:
                    X_list.append(np.full(len(subj_data), np.nan))
            
            X_full = np.concatenate(X_list)
            
            # Standardization: single predictor
            if nme in no_standardize_models:
                X_cache = X_full.copy()
            else:
                X_mean = np.nanmean(X_full)
                X_std = np.nanstd(X_full)
                if X_std == 0:
                    X_std = 1.0
                X_cache = (X_full - X_mean) / X_std
            
            valid_idx = ~(np.isnan(X_cache) | np.isnan(combined_responses_ppc))
            X = X_cache[valid_idx][:, np.newaxis]
            y = combined_responses_ppc[valid_idx]
            subj_idx = combined_subject_idx_ppc[valid_idx]
            n_features = 1
        
        n_obs = len(y)
        
        # --------- BUILD AND FIT MODEL ---------
        with pm.Model() as hmodel_ppc:
            mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=1)
            sigma_alpha = pm.Exponential('sigma_alpha', 1)
            alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=n_subjects_ppc)
            alpha = mu_alpha + sigma_alpha * alpha_offset
            
            if n_features == 1:
                # Single feature model
                mu_beta = pm.Normal('mu_beta', mu=0, sigma=1)
                sigma_beta = pm.Exponential('sigma_beta', 1)
                beta_offset = pm.Normal('beta_offset', mu=0, sigma=1, shape=n_subjects_ppc)
                beta = mu_beta + sigma_beta * beta_offset
                eta = alpha[subj_idx] + beta[subj_idx] * X[:, 0]
            else:
                # Multi-feature model with random slopes and fixed interactions
                mu_beta_feat = pm.Normal('mu_beta_feat', mu=0, sigma=1, shape=n_features)
                sigma_beta_feat = pm.Exponential('sigma_beta_feat', 1, shape=n_features)
                beta_offset = pm.Normal('beta_offset', mu=0, sigma=1, shape=(n_subjects_ppc, n_features))
                beta = mu_beta_feat[np.newaxis, :] + sigma_beta_feat[np.newaxis, :] * beta_offset
                
                # Interaction terms (pairwise products) - FIXED GROUP-LEVEL ONLY
                # NOTE: WWS × BES is excluded (mutually exclusive variables)
                interactions = np.array([
                    X[:, i] * X[:, j]
                    for i, j in interaction_pairs
                ]).T
                
                n_interactions = interactions.shape[1]
                beta_interactions = pm.Normal('beta_interactions', mu=0, sigma=1, shape=n_interactions)
                eta = (alpha[subj_idx] + 
                       pm.math.sum(beta[subj_idx] * X, axis=1) +
                       pm.math.sum(beta_interactions * interactions, axis=1))
            
            y_obs = pm.Bernoulli('y_obs', logit_p=eta, observed=y)
            
            idata_ppc = pm.sample(1000, tune=1000, random_seed=42,
                                  return_inferencedata=True,
                                  progressbar=True, chains=4, cores=4,
                                  target_accept=0.9)
        
        # Compute posterior predictive
        with hmodel_ppc:
            ppc_samples_obj = pm.sample_posterior_predictive(idata_ppc, random_seed=42)
        
        # Store PPC results
        ppc_results.append({
            'model_name': nme,
            'display_name': display_name,
            'ppc_samples': ppc_samples_obj.posterior_predictive['y_obs'].values,
            'observed_data': idata_ppc.observed_data['y_obs'].values
        })
        
        print(f"  Model fitted and PPC computed.")
    
    # --------- CREATE PUBLICATION-READY PPC PLOTS ---------
    print("\n" + "="*60)
    print("Generating Publication-Ready PPC Plots")
    print("="*60)
    
    for ppc_data in ppc_results:
        model_name = ppc_data['model_name']
        display_name = ppc_data['display_name']
        ppc_samples = ppc_data['ppc_samples']  # Shape: (chains, draws, n_obs)
        observed_data = ppc_data['observed_data']
        
        print(f"\nPlotting: {display_name}")
        
        # Reshape PPC samples to (n_samples, n_obs)
        n_chains, n_draws, n_obs = ppc_samples.shape
        ppc_samples_flat = ppc_samples.reshape(n_chains * n_draws, n_obs)
        
        # Compute predicted probabilities
        ppc_prob = ppc_samples_flat.mean(axis=0)
        
        # Create publication-ready plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        
        # ===== LEFT PANEL: Empirical vs Predicted =====
        ax = axes[0]
        
        # Separate by observed response
        predicted_when_choose = ppc_prob[observed_data == 1]
        predicted_when_avoid = ppc_prob[observed_data == 0]
        
        # Violin plots
        parts_choose = ax.violinplot([predicted_when_choose], positions=[0], widths=0.7,
                                      showmeans=True, showextrema=False)
        parts_avoid = ax.violinplot([predicted_when_avoid], positions=[1], widths=0.7,
                                    showmeans=True, showextrema=False)
        
        # Color the violin plots
        for pc in parts_choose['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.7)
        for pc in parts_avoid['bodies']:
            pc.set_facecolor('coral')
            pc.set_alpha(0.7)
        
        # Overlay individual points with jitter
        x_jitter_choose = np.random.normal(0, 0.04, size=len(predicted_when_choose))
        x_jitter_avoid = np.random.normal(1, 0.04, size=len(predicted_when_avoid))
        
        ax.scatter(x_jitter_choose, predicted_when_choose, alpha=0.3, s=20, color='steelblue')
        ax.scatter(x_jitter_avoid, predicted_when_avoid, alpha=0.3, s=20, color='coral')
        
        # Labels and formatting
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Forage', 'Wait'], fontsize=28)
        ax.set_ylabel('Predicted Probability', fontsize=28)
        ax.set_ylim([-0.05, 1.05])
        ax.tick_params(axis='y', labelsize=24)
        ax.tick_params(bottom=True, left=True, size=5, direction='in')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # ===== RIGHT PANEL: Distribution of PPC samples =====
        ax = axes[1]
        
        # Plot histogram of predicted values conditioned on response
        ax.hist(predicted_when_choose, bins=20, alpha=0.6, label='Observed: Forage', 
                color='steelblue', edgecolor='black', linewidth=1.5)
        ax.hist(predicted_when_avoid, bins=20, alpha=0.6, label='Observed: Wait',
                color='coral', edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Predicted Probability', fontsize=28)
        ax.set_ylabel('Frequency', fontsize=28)
        ax.tick_params(axis='x', labelsize=24)
        ax.tick_params(axis='y', labelsize=24)
        ax.tick_params(bottom=True, left=True, size=5, direction='in')
        ax.legend(fontsize=22, loc='upper right', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Title
        fig.suptitle(f'Posterior Predictive Check: {display_name}', 
                     fontsize=32, y=1.00)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # Save figure
        sanitized_name = display_name.replace(' ', '_').replace('$', '').replace('\\', '').replace('{', '').replace('}', '')
        fig_path = path + f'RESULTS/PPC_{sanitized_name}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: RESULTS/PPC_{sanitized_name}.png")
        
        plt.show()
        plt.close()
    
    print("\nPosterior Predictive Check plots complete.")

# %% MAIN: Model comparison using PSIS-LOO
""" Model Comparison using PSIS-LOO """
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Model Comparison via PSIS-LOO Cross-Validation")
    print("="*60)
    
    # Compute LOO for each model
    loo_values = []
    loo_diffs = []
    
    for model_idx, (idata, info) in enumerate(zip(idata_list, model_data_info)):
        print(f"\nComputing LOO for model {model_idx + 1}/{len(mdlName)}: {info['model_name']}")
        
        # Compute LOO using arviz
        loo_result = az.loo(idata)
        loo_values.append(loo_result.elpd_loo)
        
        print(f"  elpd_loo: {loo_result.elpd_loo:.2f}")
        print(f"  pareto_k: {loo_result.pareto_k.mean():.3f}")
    
    # Normalize LOO values (relative to best model)
    loo_values = np.array(loo_values)
    loo_normalized = loo_values - loo_values.max()  # Best model has value 0
    
    # Save LOO values
    loo_df = pd.DataFrame(loo_normalized).T
    loo_df.columns = mdlName
    loo_df.to_csv(path + 'RESULTS/fora_LOO.csv', index=False)
    
    print("\n" + "="*60)
    print("LOO Summary (normalized to best model)")
    print("="*60)
    for mdl, loo_val in zip(mdlName, loo_normalized):
        print(f"{mdl:40s}: {loo_val:10.2f}")


# %% Plot Model comparison results
""" Plotting """
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Generating plots")
    print("="*60)
    
    # Plotting
    plot_names = mdlName.copy()
    if 'multi-heuristic policy' in plot_names:
        multi_idx = plot_names.index('multi-heuristic policy')
        plot_names[multi_idx] = 'multi-feature policy'
    
    # Relabel models with ** to * for plotting
    plot_names = [name.replace('** ', '* ') for name in plot_names]
    
    # Rename expected gain naive to expected gain for plotting
    plot_names = [name.replace('expected gain naive', 'expected gain') for name in plot_names]
    plot_names = [name.replace('optimal policy values', r'$\Delta Q$-values') for name in plot_names]
    
    valu = loo_normalized
    
    # Sort by LOO values (best to worst)
    sorted_indices = np.argsort(valu)  # Negative for descending order
    sorted_names = [plot_names[i] for i in sorted_indices]
    sorted_valu = valu[sorted_indices]
    
    if condition == 0:
        name = sorted_names
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.tick_params(axis="x", labelsize=34)
        ax.tick_params(axis="y", labelsize=34)
        ax.tick_params(bottom=True, left=True, size=5, direction="in")
        bars = ax.barh(name, sorted_valu)
        ax.get_yticklabels()[-1].set_color('blue')  # Color best model label blue
        ax.set_title('PSIS-LOO Model Comparison',
                      loc='left', size=46)
        plt.xlabel("Δ ELPD (best model - model x)", fontsize=40)
        
    elif condition == 1:
        name = sorted_names
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.tick_params(axis="x", labelsize=34)
        ax.tick_params(axis="y", labelsize=34)
        ax.tick_params(bottom=True, left=True, size=5, direction="in")
        bars = ax.barh(name, sorted_valu)
        bars[-1].set_color('blue')  # Color best model bar blue
        ax.get_yticklabels()[-1].set_color('blue')  # Color best model label blue
        ax.set_title('Approach Forests',
                      loc='left', size=46)
        plt.xlabel("Δ ELPD (best model - model x)", fontsize=40)
        
    elif condition == 2:
        name = sorted_names
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.tick_params(axis="x", labelsize=34)
        ax.tick_params(axis="y", labelsize=34)
        ax.tick_params(bottom=True, left=True, size=5, direction="in")
        bars = ax.barh(name, sorted_valu)
        bars[-1].set_color('blue')  # Color best model bar blue
        ax.get_yticklabels()[-1].set_color('blue')  # Color best model label blue
        ax.set_title('Avoidance Forests',
                      loc='left', size=46)
        plt.xlabel("Δ ELPD (best model - model x)", fontsize=40)
    
    plt.show()


# %% Correlation OP slope and survival
""" Correlation: Optimal Policy Coefficients vs Survival Rate """
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Optimal Policy Coefficients vs Survival Rate Analysis")
    print("="*60)
    
    # Find the optimal policy values model index
    op_idx = mdlName.index('optimal policy values')
    op_idata = idata_list[op_idx]  # Get the inference data for optimal policy model
    
    # Compute metrics per subject
    subject_metrics = []
    
    for subj_id in range(n_subjects):
        subj_data = all_data[subj_id]
        
        # Survival rate: count trials where day.thisRepN == 7 AND out_LP != 0 (across both conditions)
        # Approach forests (p/r heuristic == "['p']")
        approach_data = subj_data[subj_data["p/r heuristic"] == "['p']"]
        approach_survival = ((approach_data['day.thisRepN'] == 7) & 
                            (approach_data['out_LP'] != 0)).sum()
        
        # Avoidance forests (p/r heuristic == "['r']")
        avoidance_data = subj_data[subj_data["p/r heuristic"] == "['r']"]
        avoidance_survival = ((avoidance_data['day.thisRepN'] == 7) & 
                             (avoidance_data['out_LP'] != 0)).sum()
        
        # Total survival rate (sum of conditions)
        survival_rate = approach_survival + avoidance_survival
        
        # Extract per-subject slope coefficient from posterior
        # Reconstruct beta = mu_beta + sigma_beta * beta_offset for this subject
        try:
            mu_beta_posterior = op_idata.posterior['mu_beta'].values  # Shape: (chains, draws)
            sigma_beta_posterior = op_idata.posterior['sigma_beta'].values  # Shape: (chains, draws)
            beta_offset_posterior = op_idata.posterior['beta_offset'].values[:, :, subj_id]  # Shape: (chains, draws)
            
            # Reconstruct beta for this subject across all posterior samples
            beta_posterior = mu_beta_posterior + sigma_beta_posterior * beta_offset_posterior  # Shape: (chains, draws)
            
            # Average across all posterior samples to get posterior mean
            op_slope_coeff = beta_posterior.mean()
        except (KeyError, IndexError, TypeError):
            op_slope_coeff = np.nan
        
        subject_metrics.append({
            'subject_id': subj_id,
            'survival_rate': survival_rate,
            'op_slope_coefficient': op_slope_coeff
        })
    
    # Create DataFrame for easier manipulation
    metrics_df = pd.DataFrame(subject_metrics)
    
    # Remove rows with NaN values
    metrics_clean = metrics_df.dropna()
    
    print(f"\nSubjects with complete data: {len(metrics_clean)}/{n_subjects}")
    print(f"\nSurvival Rate (Mean ± SD): {metrics_clean['survival_rate'].mean():.3f} ± {metrics_clean['survival_rate'].std():.3f}")
    print(f"OP Slope Coefficient (Mean ± SD): {metrics_clean['op_slope_coefficient'].mean():.3f} ± {metrics_clean['op_slope_coefficient'].std():.3f}")
    
    # Compute correlations
    from scipy.stats import pearsonr, spearmanr
    
    if len(metrics_clean) > 2:
        # Pearson correlation
        r_pearson, p_pearson = pearsonr(metrics_clean['survival_rate'], metrics_clean['op_slope_coefficient'])
        # Spearman correlation
        r_spearman, p_spearman = spearmanr(metrics_clean['survival_rate'], metrics_clean['op_slope_coefficient'])
        
        print(f"\n{'='*60}")
        print("CORRELATION ANALYSIS")
        print(f"{'='*60}")
        print(f"\nSurvival Rate vs OP Slope Coefficient:")
        print(f"  Pearson r = {r_pearson:.4f}, p = {p_pearson:.4f}")
        print(f"  Spearman ρ = {r_spearman:.4f}, p = {p_spearman:.4f}")
        
        # Create scatter plot with regression line and statistics
        x = metrics_clean['op_slope_coefficient'].values
        y = metrics_clean['survival_rate'].values
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
        
        # Plot scatter points
        ax.scatter(x, y, s=80, alpha=0.7)
        
        # Compute linear fit using polyfit
        coeffs = np.polyfit(x, y, 1)  # Linear fit: y = slope * x + intercept
        fit_line = np.poly1d(coeffs)
        x_line = np.array([x.min(), x.max()])
        y_line = fit_line(x_line)
        
        # Plot fit line in red
        ax.plot(x_line, y_line, color='red', linewidth=2)
        
        # Add grid lines
        ax.grid(True, alpha=0.3, linestyle='--')
        
        ax.tick_params(bottom=True, left=True, size=5, direction="in")
        ax.tick_params(axis="x", labelsize=22)
        ax.tick_params(axis="y", labelsize=22)
        plt.xlabel(r"$\beta_1$ (slope) coefficients for " + r'$\Delta Q$-values', fontsize=28)
        ax.set_ylabel('No. of survived forests', fontsize=28, labelpad=-1)
        
        # Add correlation statistics in italic inside the plot
        stats_text = f"r = {r_pearson:.2f}, p = {p_pearson:.2f}"
        ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, 
                fontsize=20, verticalalignment='bottom', horizontalalignment='right', style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(path + 'RESULTS/op_slope_vs_survival.png', dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: RESULTS/op_slope_vs_survival.png")
        plt.show()
        
        # Save correlation results
        corr_results = pd.DataFrame({
            'Method': ['Pearson', 'Spearman'],
            'Correlation': [r_pearson, r_spearman],
            'P_value': [p_pearson, p_spearman],
            'N': [len(metrics_clean), len(metrics_clean)]
        })
        corr_results.to_csv(path + 'RESULTS/op_slope_survival_correlation.csv', index=False)
        print(f"Correlation results saved to: RESULTS/op_slope_survival_correlation.csv")
    else:
        print(f"\nWarning: Insufficient data for correlation (n={len(metrics_clean)}, need > 2)")


# %%
# %%
# %%
# %%
# %% Diagnostics: Separation per condition
""" Data Diagnostics: Check for Separation Issues per Condition and Feature """
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Separation Diagnostics: Feature Predictiveness by Condition")
    print("="*60)
    
    # Load full dataset for diagnostics
    all_data_diag = []
    all_responses_diag = []
    all_conditions_diag = []
    
    for fle in glob.glob(path + "DATA_clean/test_data.*.CAT.csv"):
        sbj = fle[len(path+"DATA_clean/test_data."):-len(".CAT.csv")]
        dt = pd.read_csv(path + "DATA_clean/test_data." + sbj + ".CAT" + ".csv")
        dt = dt[dt['foraging T/F NaNs'].isnull() == False]
        
        # Encode condition
        cond_array = [(1 if row["p/r heuristic"] == "['r']" else 0) for _, row in dt.iterrows()]
        
        all_data_diag.append(dt)
        all_responses_diag.append(np.array(dt["fora_response"]))
        all_conditions_diag.append(np.array(cond_array))
    
    features_to_check = [
        '* $\\mathit{p}$ gain',
        '* $\\mathit{r}$ predator',
        '** wait when safe',
        '** binary energy state',
        'optimal policy values'
    ]
    
    print("\nFeature variation and predictiveness per condition:")
    print("(High variance separation → potential fitting issues)\n")
    
    for feat in features_to_check:
        print(f"{feat}:")
        for cond_name, cond_val in [("Approach (p)", 1), ("Avoidance (r)", 0)]:
            feat_vals_cond = []
            resp_vals_cond = []
            
            for dt, resp, conds in zip(all_data_diag, all_responses_diag, all_conditions_diag):
                if feat in dt.columns:
                    mask = (conds == cond_val) & ~pd.isna(dt[feat])
                    feat_vals_cond.extend(dt[mask][feat].values)
                    resp_vals_cond.extend(resp[mask])
            
            if len(feat_vals_cond) > 0:
                feat_vals_cond = np.array(feat_vals_cond)
                resp_vals_cond = np.array(resp_vals_cond)
                
                # Check for separation: correlation between feature and response
                if len(np.unique(feat_vals_cond)) > 1:
                    # Skip constant features
                    from scipy.stats import pointbiserialr
                    corr, p_val = pointbiserialr(resp_vals_cond, feat_vals_cond)
                    unique_vals = len(np.unique(feat_vals_cond))
                    
                    print(f"  {cond_name:15s}: n={len(feat_vals_cond):4d}, corr={corr:7.3f}, "
                          f"p<.001={p_val<0.001}, unique_vals={unique_vals}")
                else:
                    print(f"  {cond_name:15s}: CONSTANT (no variance)")
        print()


# %% MAIN: Condition modulation on MH and OP
""" Unified Hierarchical Model: Full Moderation of Condition and Blocks """
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Unified Model: MH + OP Features (Standardized, with Condition Effects)")
    print("="*60)
    print("(All conditions pooled; Continuous variables standardized; OP globally standardized)")
    print("(Condition is a main effect + interactions with features)")
    print("="*60)
    
    # Define features for the unified condition model
    unified_features = [
        '* $\\mathit{p}$ gain',
        '* $\\mathit{r}$ predator',
        '** wait when safe',
        '** binary energy state',
        'optimal policy values'
    ]
    
    # Load all data (full dataset, no condition filtering - CRITICAL for avoiding separation)
    all_data_unified = []
    all_responses_unified = []
    subject_ids_unified = []
    all_conditions_unified = []  # 1 = approach (p), 0 = avoidance (r)
    all_block_idx_unified = []  # Block index per observation
    
    # Decide whether to residualize OP with respect to MH features
    residualized = False
    
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


        # ------------------------------------------------------------------------------
        # BLOCK CREATION (PRESERVE TRUE ORDER)
        # ------------------------------------------------------------------------------

        # Previous miniblock value per subject
        prev_mb = dt.groupby("Subject_ID")["miniblock.thisN"].shift()

        # Detect FIRST 0 of each sequence
        dt["block_start"] = (
            (dt["miniblock.thisN"] == 0) & (prev_mb != 0)
        ).astype(int)

        # Count blocks per subject
        dt["block_idx"] = dt.groupby("Subject_ID")["block_start"].cumsum() -1
        
        # Filter but keep BOTH conditions (this is key - no subsetting!)
        dt = dt[dt['foraging T/F NaNs'].isnull() == False]
        dt = dt.reset_index(drop=True)# %% MAIN: Condition Modulation Effect on MH and OP

        # Encode condition: 1 for approach (p), 0 for avoidance (r)
        cond_array = np.array([1 if row["p/r heuristic"] == "['r']" else 0 
                               for _, row in dt.iterrows()])
        
        # Extract block_idx
        block_idx_array = np.array(dt["block_idx"])
        
        all_data_unified.append(dt)
        all_responses_unified.append(np.array(dt["fora_response"]))
        all_conditions_unified.append(cond_array)
        all_block_idx_unified.append(block_idx_array)
        subject_ids_unified.append(np.full(len(dt), s_idx))
        s_idx += 1
    
    n_subjects_unified = len(all_data_unified)
    
    # Prepare data for the unified model
    X_list_unified = []
    for feat in unified_features:
        feat_data = []
        for subj_data in all_data_unified:
            if feat in subj_data.columns:
                feat_data.append(np.array(subj_data[feat]))
            else:
                feat_data.append(np.full(len(subj_data), np.nan))
        X_list_unified.append(np.concatenate(feat_data))
    
    # Add condition and block_idx as features
    cond_unified = np.concatenate(all_conditions_unified)
    X_list_unified.append(cond_unified)
    
    block_idx_unified = np.concatenate(all_block_idx_unified)
    X_list_unified.append(block_idx_unified)
    
    X_full_unified = np.array(X_list_unified).T
    responses_unified = np.concatenate(all_responses_unified)
    subj_idx_unified = np.concatenate(subject_ids_unified)
    
    # Create validity mask: no NaNs in features or responses
    valid_idx_unified = ~(np.isnan(X_full_unified).any(axis=1) | np.isnan(responses_unified))
    X_unified = X_full_unified[valid_idx_unified]
    y_unified = responses_unified[valid_idx_unified]
    subj_idx_unified = subj_idx_unified[valid_idx_unified]
    
    n_features_unified = len(unified_features)  # 5 main features (no condition as main effect)
    n_obs_unified = len(y_unified)
    
    print(f"\nUnified model data (pooled across conditions): {n_obs_unified} observations from {n_subjects_unified} subjects")
    
    # -------------------------------------------------------------------------------------
    # ===== RESIDUALIZE OP IF residualized is TRUE =====
    if residualized:
        print("\n" + "="*60)
        print("Residualizing Optimal Policy (ΔQ) w.r.t. MH Features")
        print("="*60)

        # Extract MH predictors (already concatenated in X_unified)
        X_mh = X_unified[:, 0:4]  # p_gain, predator, WWS, BES
        op_raw = X_unified[:, 4]

        # Add intercept for residualization
        X_mh_with_intercept = np.column_stack([np.ones(len(X_mh)), X_mh])

        # Solve linear regression: OP ~ MH features
        beta_resid = np.linalg.lstsq(X_mh_with_intercept, op_raw, rcond=None)[0]

        # Predicted OP from MH
        op_pred = X_mh_with_intercept @ beta_resid

        # Residualized OP
        op_resid = op_raw - op_pred

        # Replace OP column with residualized version
        X_unified[:, 4] = op_resid

        print(f"Residualization complete:")
        print(f"  Variance original OP: {np.var(op_raw):.6f}")
        print(f"  Variance residualized OP: {np.var(op_resid):.6f}")
        print(f"  Correlation OP vs residualized OP: {np.corrcoef(op_raw, op_resid)[0,1]:.4f}")
    # -------------------------------------------------------------------------------------

    # ===== STANDARDIZE CONTINUOUS VARIABLES (FLEXIBLE: OP OR RESIDUALIZED OP) =====
    print("\n" + "="*60)
    print("Standardizing Continuous Variables")
    print("="*60)

    # ------------------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------------------
    USE_RESIDUALIZED_OP = True  # <-- toggle this depending on pipeline

    # Feature indices (must match your design matrix!)
    IDX_P_GAIN = 0
    IDX_PREDATOR = 1
    IDX_WWS = 2
    IDX_BES = 3
    IDX_OP = 4
    IDX_CONDITION = 5
    IDX_BLOCK = 6

    # Continuous variables to standardize
    continuous_vars = {
        'p_gain': IDX_P_GAIN,
        'predator': IDX_PREDATOR,
        'OP_residual' if USE_RESIDUALIZED_OP else 'OP': IDX_OP,
        'block_idx': IDX_BLOCK
    }

    # Store parameters (useful for plotting / reproducibility)
    standardization_params = {}

    # ------------------------------------------------------------------------------
    # STANDARDIZE CONTINUOUS VARIABLES
    # ------------------------------------------------------------------------------
    for var_name, idx in continuous_vars.items():
        var_data = X_unified[:, idx]
        
        var_mean = np.nanmean(var_data)
        var_std = np.nanstd(var_data)
        
        # Safety check (important!)
        if var_std == 0:
            raise ValueError(f"Standard deviation is zero for {var_name} — cannot standardize.")
        
        # Standardize
        X_unified[:, idx] = (var_data - var_mean) / var_std
        
        # Store parameters
        standardization_params[var_name] = {
            'mean': var_mean,
            'std': var_std
        }
        
        print(f"\n{var_name} (global standardization):")
        print(f"  Original mean: {var_mean:.4f}, Original std: {var_std:.4f}")
        print(f"  Standardized (check): mean = {np.nanmean(X_unified[:, idx]):.6f}, std = {np.nanstd(X_unified[:, idx]):.6f}")

    # ------------------------------------------------------------------------------
    # NON-STANDARDIZED VARIABLES (EXPLICIT FOR CLARITY)
    # ------------------------------------------------------------------------------
    print("\nNon-standardized variables:")
    print("  wait_when_safe (binary)")
    print("  binary_energy_state (binary)")
    print("  condition (binary)")

    # ------------------------------------------------------------------------------
    # EXTRACT MODERATOR VARIABLES (AFTER STANDARDIZATION)
    # ------------------------------------------------------------------------------
    condition_for_interactions = X_unified[:, IDX_CONDITION].copy()
    block_idx_for_interactions = X_unified[:, IDX_BLOCK].copy()

    # Build hierarchical PyMC model: UNIFIED APPROACH
    # Main effects have random slopes per subject (captures individual differences in decision-making)
    # Includes condition as main effect (with random slopes) plus interactions with features (fixed)
    # All continuous variables (p_gain, predator, OP) are standardized globally
    with pm.Model() as model_unified:
        # Group-level hyperpriors for intercept
        mu_alpha_u = pm.Normal('mu_alpha_u', mu=0, sigma=1)
        sigma_alpha_u = pm.Exponential('sigma_alpha_u', 1)
        
        # Subject-level intercept parameters (non-centered parameterization)
        alpha_offset_u = pm.Normal('alpha_offset_u', mu=0, sigma=1, shape=n_subjects_unified)
        alpha_u = mu_alpha_u + sigma_alpha_u * alpha_offset_u
        
        # ===== MAIN EFFECTS (RANDOM SLOPES) =====
        # Each subject has their own slope for all 7 features (MH features + OP + condition + block_idx)
        # Continuous variables (p_gain, predator, OP, block_idx) are standardized for comparable magnitudes
        # This captures individual differences in how strongly each factor influences behavior
        n_features_with_condition = n_features_unified + 2  # 5 features + condition + block_idx
        mu_beta_u = pm.Normal('mu_beta_u', mu=0, sigma=1, shape=n_features_with_condition)
        sigma_beta_u = pm.Exponential('sigma_beta_u', 1, shape=n_features_with_condition)
        
        # Subject deviations for random slopes
        beta_offset_u = pm.Normal('beta_offset_u', mu=0, sigma=1, 
                                  shape=(n_subjects_unified, n_features_with_condition))
        beta_u = mu_beta_u[np.newaxis, :] + sigma_beta_u[np.newaxis, :] * beta_offset_u
        
        # ===== CONDITION INTERACTIONS (FIXED GROUP-LEVEL) =====
        # Tests how condition modulates each feature's influence on choice
        # These are hypothesis tests, not random effects
        # Note: OP column contains raw optimal policy values (standardized globally)
        # Note: Continuous variables (p_gain, predator, OP) are standardized for comparable scales
        interactions_condition = np.array([
            X_unified[:, 0] * condition_for_interactions,   # p_gain (standardized) × condition
            X_unified[:, 1] * condition_for_interactions,   # predator (standardized) × condition
            X_unified[:, 2] * condition_for_interactions,   # wait_when_safe × condition
            X_unified[:, 3] * condition_for_interactions,   # binary_energy_state × condition
            X_unified[:, 4] * condition_for_interactions,   # OP (standardized) × condition
        ]).T  # Shape: (n_obs, 5)
        
        n_interactions_condition = interactions_condition.shape[1]
        
        # ===== BLOCK INDEX INTERACTIONS (FIXED GROUP-LEVEL) =====
        # Tests how block number modulates each feature's influence on choice
        # These are hypothesis tests, not random effects
        interactions_block_idx = np.array([
            X_unified[:, 0] * block_idx_for_interactions,   # p_gain (standardized) × block_idx
            X_unified[:, 1] * block_idx_for_interactions,   # predator (standardized) × block_idx
            X_unified[:, 2] * block_idx_for_interactions,   # wait_when_safe × block_idx
            X_unified[:, 3] * block_idx_for_interactions,   # binary_energy_state × block_idx
            X_unified[:, 4] * block_idx_for_interactions,   # OP (standardized) × block_idx
        ]).T  # Shape: (n_obs, 5)
        
        n_interactions_block_idx = interactions_block_idx.shape[1]
        
        # Fixed group-level coefficients for condition modulation
        beta_interactions_condition = pm.Normal('beta_mod_condition', mu=0, sigma=1, shape=n_interactions_condition)
        
        # Fixed group-level coefficients for block_idx modulation
        beta_interactions_block_idx = pm.Normal('beta_mod_block_idx', mu=0, sigma=1, shape=n_interactions_block_idx)
        
        # ===== FEATURE-FEATURE INTERACTIONS (FIXED GROUP-LEVEL) =====
        # Tests synergistic effects between p_gain and predator
        interactions_feature_feature = np.array([
            X_unified[:, 0] * X_unified[:, 1],   # p_gain (standardized) × predator (standardized)
        ]).T  # Shape: (n_obs, 1)
        
        n_interactions_feature_feature = interactions_feature_feature.shape[1]
        
        # ===== THREE-WAY INTERACTION (FIXED GROUP-LEVEL) =====
        # Tests how condition modulates the p_gain × predator synergy
        interactions_three_way = np.array([
            X_unified[:, 0] * X_unified[:, 1] * condition_for_interactions,   # p_gain × predator × condition
        ]).T  # Shape: (n_obs, 1)
        
        n_interactions_three_way = interactions_three_way.shape[1]
        
        # Fixed group-level coefficients for feature-feature interactions
        beta_interactions_feature_feature = pm.Normal('beta_mod_feature_feature', mu=0, sigma=1, shape=n_interactions_feature_feature)
        
        # Fixed group-level coefficients for three-way interactions
        beta_interactions_three_way = pm.Normal('beta_mod_three_way', mu=0, sigma=1, shape=n_interactions_three_way)
        
        # ===== LINEAR PREDICTOR =====
        # intercept + main effects (random slopes, includes condition + block_idx) + 
        # feature×condition modulation (fixed) + feature×block_idx modulation (fixed) +
        # feature×feature interactions (fixed) + three-way interactions (fixed)
        eta_u = (alpha_u[subj_idx_unified] + 
                 pm.math.sum(beta_u[subj_idx_unified] * X_unified, axis=1) +
                 pm.math.sum(beta_interactions_condition * interactions_condition, axis=1) +
                 pm.math.sum(beta_interactions_block_idx * interactions_block_idx, axis=1) +
                 pm.math.sum(beta_interactions_feature_feature * interactions_feature_feature, axis=1) +
                 pm.math.sum(beta_interactions_three_way * interactions_three_way, axis=1))
        
        # Likelihood
        y_obs_u = pm.Bernoulli('y_obs_u', logit_p=eta_u, observed=y_unified)
        
        # Sample from posterior
        print("\nSampling from posterior for unified model...")
        idata_unified = pm.sample(1000, tune=1000, random_seed=42,
                                  return_inferencedata=True,
                                  progressbar=True, chains=4, cores=4,
                                  target_accept=0.9)
        
        # Compute log_likelihood for LOO calculation
        idata_unified = pm.compute_log_likelihood(idata_unified)
    
    # Compute posterior predictive
    with model_unified:
        ppc_u = pm.sample_posterior_predictive(idata_unified, random_seed=42)
    
    pred_prob_u = ppc_u.posterior_predictive['y_obs_u'].mean(axis=(0, 1)).values
    
    # Compute LOO for comparison
    loo_unified = az.loo(idata_unified)
    print(f"\nUnified Model LOO Results:")
    print(f"  ELPD LOO: {loo_unified.elpd_loo:.2f}")
    print(f"  Pareto K (mean): {loo_unified.pareto_k.mean():.3f}")
    
    # Extract and display posterior summaries
    print(f"\n{'='*60}")
    print("Unified Model Posterior Summary")
    print(f"{'='*60}")
    
    # Display random intercepts
    print("\nGroup-level intercept:")
    print(f"  μ(α) = {idata_unified.posterior['mu_alpha_u'].values.mean():.4f}")
    print(f"  σ(α) = {idata_unified.posterior['sigma_alpha_u'].values.mean():.4f}")
    
    # Display main effect hyperpriors
    print("\nMain effects (group-level hyperpriors, random slopes per subject):")
    print("(Note: Continuous variables (p_gain, predator, OP) are globally standardized; block_idx standardized)")
    mu_beta_u_vals = idata_unified.posterior['mu_beta_u'].values.mean(axis=(0, 1))
    sigma_beta_u_vals = idata_unified.posterior['sigma_beta_u'].values.mean(axis=(0, 1))
    
    # Note: optimal policy values is standardized globally
    # Continuous variables are standardized: p_gain, predator, OP, block_idx
    feature_names_unified = ['p_gain (standardized)', 'predator (standardized)', 'wait_when_safe', 'binary_energy_state', 'OP (standardized)', 'condition', 'block_idx (standardized)']
    for i, fname in enumerate(feature_names_unified):
        print(f"  {fname:38s}: mu = {mu_beta_u_vals[i]:8.4f}, sigma = {sigma_beta_u_vals[i]:8.4f}")
    
    # Display condition modulation effects (the key hypothesis tests)
    print("\nCondition Modulation Effects (fixed group-level; tests how condition changes effect):")
    print("(Note: Standardized continuous variables; OP globally standardized)")
    beta_mod_condition_vals = idata_unified.posterior['beta_mod_condition'].values.mean(axis=(0, 1))
    condition_mod_names = ['p_gain x condition', 'predator x condition', 'wait_when_safe x condition', 
                           'binary_energy_state x condition', 'OP x condition']
    for i, mname in enumerate(condition_mod_names):
        print(f"  {mname:38s}: {beta_mod_condition_vals[i]:8.4f}")
    
    # Display block_idx modulation effects
    print("\nBlock Number Modulation Effects (fixed group-level; tests how block number changes effect):")
    print("(Note: Standardized continuous variables and block_idx; OP globally standardized)")
    beta_mod_block_idx_vals = idata_unified.posterior['beta_mod_block_idx'].values.mean(axis=(0, 1))
    block_idx_mod_names = ['p_gain x block_idx', 'predator x block_idx', 'wait_when_safe x block_idx', 
                           'binary_energy_state x block_idx', 'OP x block_idx']
    for i, bname in enumerate(block_idx_mod_names):
        print(f"  {bname:38s}: {beta_mod_block_idx_vals[i]:8.4f}")
    
    # Display feature-feature interaction effects
    print("\nFeature-Feature Interaction Effects (fixed group-level; tests synergistic effects):")
    print("(Note: Standardized continuous variables)")
    beta_mod_feature_feature_vals = idata_unified.posterior['beta_mod_feature_feature'].values.mean(axis=(0, 1))
    feature_feature_mod_names = ['p_gain x predator']
    for i, ffname in enumerate(feature_feature_mod_names):
        print(f"  {ffname:38s}: {beta_mod_feature_feature_vals[i]:8.4f}")
    
    # Display three-way interaction effects
    print("\nThree-Way Interaction Effects (fixed group-level; tests how condition modulates synergy):")
    print("(Note: Standardized continuous variables)")
    beta_mod_three_way_vals = idata_unified.posterior['beta_mod_three_way'].values.mean(axis=(0, 1))
    three_way_mod_names = ['p_gain x predator x condition']
    for i, twname in enumerate(three_way_mod_names):
        print(f"  {twname:38s}: {beta_mod_three_way_vals[i]:8.4f}")
    
    # Save unified model results
    loo_unified_val = loo_unified.elpd_loo
    unified_model_results = pd.DataFrame({
        'Model': ['Unified: MH + OP (Standardized Continuous, Globally Standardized OP)'],
        'ELPD_LOO': [loo_unified_val],
        'Pareto_K_mean': [loo_unified.pareto_k.mean()],
        'N_obs': [n_obs_unified],
        'N_subjects': [n_subjects_unified]
    })
    unified_model_results.to_csv(path + 'RESULTS/unified_condition_model_results.csv', index=False)
    print(f"\nUnified model results saved to: RESULTS/unified_condition_model_results.csv")
    
    # Compare to original models
    print(f"\n{'='*60}")
    print("Comparison: Unified Model vs Original Models (LOO)")
    print(f"{'='*60}")
    print(f"\nUnified model ELPD LOO: {loo_unified_val:.2f}")
    if 'loo_values' in dir():
        print(f"Best single-predictor model: {loo_values.max():.2f}")
        print(f"Multi-heuristic policy model: {loo_values[mdlName.index('multi-heuristic policy')]:.2f}")
        print(f"\nDifference (unified - multi-heuristic): {loo_unified_val - loo_values[mdlName.index('multi-heuristic policy')]:.2f}")
        print(f"  (Negative = unified is worse; Positive = unified is better)")
    
# %% Full modulation model model visualization
# ===== FOREST PLOT FULL =====
print(f"\n{'='*60}")
print("Generating Forest Plot (Publication Quality)")
print(f"{'='*60}")

# Extract posterior samples for main effects and interactions
mu_beta_samples = idata_unified.posterior['mu_beta_u'].values  # Shape: (chains, draws, n_features)
beta_interactions_condition_samples = idata_unified.posterior['beta_mod_condition'].values  # Shape: (chains, draws, n_interactions_condition)
beta_interactions_block_idx_samples = idata_unified.posterior['beta_mod_block_idx'].values  # Shape: (chains, draws, n_interactions_block_idx)
beta_interactions_feature_feature_samples = idata_unified.posterior['beta_mod_feature_feature'].values  # Shape: (chains, draws, n_interactions_feature_feature)
beta_interactions_three_way_samples = idata_unified.posterior['beta_mod_three_way'].values  # Shape: (chains, draws, n_interactions_three_way)

# Flatten to (n_samples, n_params)
mu_beta_flat = mu_beta_samples.reshape(-1, mu_beta_samples.shape[-1])  # (chains*draws, 7)
beta_interactions_condition_flat = beta_interactions_condition_samples.reshape(-1, beta_interactions_condition_samples.shape[-1])  # (chains*draws, 5)
beta_interactions_block_idx_flat = beta_interactions_block_idx_samples.reshape(-1, beta_interactions_block_idx_samples.shape[-1])  # (chains*draws, 5)
beta_interactions_feature_feature_flat = beta_interactions_feature_feature_samples.reshape(-1, beta_interactions_feature_feature_samples.shape[-1])  # (chains*draws, 1)
beta_interactions_three_way_flat = beta_interactions_three_way_samples.reshape(-1, beta_interactions_three_way_samples.shape[-1])  # (chains*draws, 1)

# Combine all parameters for forest plot
all_params = np.hstack([mu_beta_flat, beta_interactions_condition_flat, beta_interactions_block_idx_flat, beta_interactions_feature_feature_flat, beta_interactions_three_way_flat])  # (chains*draws, 19)

# Parameter names for forest plot (with improved labels)
forest_param_names = [
    r'$\mathit{p}$(gain)',
    r'$\mathit{r}$(threat)',
    'wait when safe',
    'binary energy',
    r'$\Delta Q$-values',
    'avoidance condition',
    'block number',
    r'$\mathit{p}$(gain) $\times$ condition',
    r'$\mathit{r}$(threat) $\times$ condition',
    'wait when safe ' + r'$\times$' + ' condition',
    'binary energy ' + r'$\times$' + ' condition',
    r'$\Delta Q$-values $\times$ condition',
    r'$\mathit{p}$(gain) $\times$ block number',
    r'$\mathit{r}$(threat) $\times$ block number',
    'wait when safe ' + r'$\times$' + ' block number',
    'binary energy ' + r'$\times$' + ' block number',
    r'$\Delta Q$-values $\times$ block number',
    r'$\mathit{p}$(gain) $\times$ $\mathit{r}$(threat)',
    r'$\mathit{p}$(gain) $\times$ $\mathit{r}$(threat) $\times$ condition'
]

if residualized:
    forest_param_names = [
        r'$\mathit{p}$(gain)',
        r'$\mathit{r}$(threat)',
        'wait when safe',
        'binary energy',
        r'$\Delta Q$-values residualized',
        'avoidance condition',
        'block number',
        r'$\mathit{p}$(gain) $\times$ condition',
        r'$\mathit{r}$(threat) $\times$ condition',
        'wait when safe ' + r'$\times$' + ' condition',
        'binary energy ' + r'$\times$' + ' condition',
        r'$\Delta Q$-values residualized $\times$ condition',
        r'$\mathit{p}$(gain) $\times$ block number',
        r'$\mathit{r}$(threat) $\times$ block number',
        'wait when safe ' + r'$\times$' + ' block number',
        'binary energy ' + r'$\times$' + ' block number',
        r'$\Delta Q$-values residualized $\times$ block number',
        r'$\mathit{p}$(gain) $\times$ $\mathit{r}$(threat)',
        r'$\mathit{p}$(gain) $\times$ $\mathit{r}$(threat) $\times$ condition'
    ]

# Compute posterior means and 95% HDI credible intervals
posterior_means = np.mean(all_params, axis=0)

# Compute 95% HDI
hdi_lower = np.percentile(all_params, 2.5, axis=0)
hdi_upper = np.percentile(all_params, 97.5, axis=0)

# Sort all parameters by effect size (highest positive to highest negative)
# This is the standard forest plot ordering convention
sort_idx = np.argsort(posterior_means)[::-1]

posterior_means_sorted = posterior_means[sort_idx]
hdi_lower_sorted = hdi_lower[sort_idx]
hdi_upper_sorted = hdi_upper[sort_idx]
param_names_sorted = [forest_param_names[i] for i in sort_idx]

# Create publication-quality forest plot
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

y_pos = np.arange(len(param_names_sorted))

# Plot credible intervals
for i, (mean, lower, upper) in enumerate(zip(posterior_means_sorted, hdi_lower_sorted, hdi_upper_sorted)):
    # Plot interval line
    ax.plot([lower, upper], [i, i], 'k-', linewidth=2, alpha=0.8)
    
    # Plot point estimate (mean)
    ax.scatter(mean, i, s=120, c='darkblue', zorder=3, edgecolors='black', linewidth=1)

# Add reference line at 0
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.6, zorder=2, label='No effect')

# Invert y-axis for top-to-bottom reading
ax.invert_yaxis()

# Formatting
ax.set_yticks(y_pos)
ax.set_yticklabels(param_names_sorted, fontsize=13)
ax.set_xlabel('Posterior Estimate (logit scale)', fontsize=14)
ax.set_title('Main Effects & Condition/Block Number Modulations\n95% Credible Intervals', 
                fontsize=15, pad=5)
if residualized:
    ax.set_title('Main Effects & Condition/Block Number Modulations\n(ΔQ Residualized) - 95% Credible Intervals', 
                    fontsize=15, pad=5)

ax.tick_params(axis='x', labelsize=12)
ax.grid(True, alpha=0.3, axis='x', linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Add legend
ax.legend(loc='lower right', fontsize=11, framealpha=0.95)

plt.tight_layout()
plt.savefig(path + 'RESULTS/unified_model_forest_plot.png', dpi=300, bbox_inches='tight')
print(f"\nForest plot saved to: RESULTS/unified_model_forest_plot.png")

# Print forest plot statistics
print(f"\n{'='*60}")
print("Forest Plot Summary (sorted by effect size)")
print(f"{'='*60}")
for name, mean, lower, upper in zip(param_names_sorted, posterior_means_sorted, hdi_lower_sorted, hdi_upper_sorted):
    crosses_zero = 'crosses 0' if lower <= 0 <= upper else 'excludes 0'
    print(f"{name:35s}: {mean:7.4f}  [{lower:7.4f}, {upper:7.4f}]  ({crosses_zero})")

# %% Modulation model effects of interest only
# ===== FOREST PLOT: KEY EFFECTS ONLY =====
print(f"\n{'='*60}")
print("Generating Forest Plot (Key Effects Only)")
print(f"{'='*60}")

# Select only key effects for main text:
# Baseline: ΔQ values, p(gain), r(threat), wait-when-safe, binary energy
# Condition effects: ΔQ×condition, p(gain)×condition, r(threat)×condition
main_text_indices = {
    'baseline': [4, 0, 1, 2, 3],  # ΔQ values, p(gain), r(threat), wait-when-safe, binary energy
    'condition': [11, 7, 8]       # ΔQ×condition, p(gain)×condition, r(threat)×condition
}

main_text_param_names = [
    r'$\Delta Q$-values',
    r'$\mathit{p}$(gain)',
    r'$\mathit{r}$(threat)',
    'wait when safe',
    'binary energy',
    r'$\Delta Q$-values $\times$ condition',
    r'$\mathit{p}$(gain) $\times$ condition',
    r'$\mathit{r}$(threat) $\times$ condition'
]

if residualized:
    main_text_param_names = [
        r'$\Delta Q$-values residualized',
        r'$\mathit{p}$(gain)',
        r'$\mathit{r}$(threat)',
        'wait when safe',
        'binary energy',
        r'$\Delta Q$-values residualized $\times$ condition',
        r'$\mathit{p}$(gain) $\times$ condition',
        r'$\mathit{r}$(threat) $\times$ condition'
    ]

# Extract parameters for main text
main_text_idx = main_text_indices['baseline'] + main_text_indices['condition']
main_text_means = posterior_means[main_text_idx]
main_text_lower = hdi_lower[main_text_idx]
main_text_upper = hdi_upper[main_text_idx]

# Sort main text parameters by effect size (highest positive to highest negative)
main_text_sort_idx = np.argsort(main_text_means)[::-1]
main_text_means_sorted = main_text_means[main_text_sort_idx]
main_text_lower_sorted = main_text_lower[main_text_sort_idx]
main_text_upper_sorted = main_text_upper[main_text_sort_idx]
main_text_param_names_sorted = [main_text_param_names[i] for i in main_text_sort_idx]

# Create figure for main text
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

# Add section headers at y positions
y_pos_list = list(range(len(main_text_idx)))

# Plot credible intervals with grouping
for i, (y_pos, mean, lower, upper) in enumerate(zip(y_pos_list, main_text_means_sorted, main_text_lower_sorted, main_text_upper_sorted)):
    # Plot interval line
    ax.plot([lower, upper], [y_pos, y_pos], 'k-', linewidth=2.5, alpha=0.8)
    
    # Plot point estimate (mean)
    ax.scatter(mean, y_pos, s=160, c='darkblue', zorder=3, edgecolors='black', linewidth=1.5)

# Add reference line at 0
ax.axvline(x=0, color='red', linestyle='--', linewidth=2.5, alpha=0.6, zorder=2, label='No effect')

# Invert y-axis for top-to-bottom reading
ax.invert_yaxis()

# Formatting with custom labels - increased to match full forest plot ratios
ax.set_yticks(y_pos_list)
ax.set_yticklabels(main_text_param_names_sorted, fontsize=15)
ax.set_xlabel('Posterior Estimate (logit scale)', fontsize=15)
ax.set_title('Core Decision Factors & Condition Modulation\n95% Credible Intervals', 
                fontsize=17, pad=12)
if residualized:
    ax.set_title('Core Decision Factors & Condition Modulation\n(ΔQ Residualized) - 95% Credible Intervals', 
                    fontsize=17, pad=12)

# # Add text annotations for section headers
# ax.text(ax.get_xlim()[0] - (ax.get_xlim()[1] - ax.get_xlim()[0])*0.15, 
#         len(main_text_indices['baseline'])/2 - 0.5, 
#         'Baseline\nDecision\nStructure',
#         fontsize=11, fontweight='bold', ha='right', va='center',
#         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# ax.text(ax.get_xlim()[0] - (ax.get_xlim()[1] - ax.get_xlim()[0])*0.15,
#         len(main_text_indices['baseline']) + len(main_text_indices['condition'])/2 - 0.5,
#         'Condition\nEffects\n(Critical)',
#         fontsize=11, fontweight='bold', ha='right', va='center',
#         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=15)
ax.grid(True, alpha=0.3, axis='x', linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Add legend
ax.legend(loc='lower right', fontsize=13, framealpha=0.95)

# Adjust layout
plt.tight_layout()
plt.savefig(path + 'RESULTS/unified_model_forest_plot_maintext.png', dpi=300, bbox_inches='tight')
print(f"\nMain forest plot saved to: RESULTS/unified_model_forest_plot_maintext.png")

# Print main text forest plot statistics (sorted by effect size)
print(f"\n{'='*60}")
print("Forest Plot Summary Key Features (sorted by effect size)")
print(f"{'='*60}")
for name, mean, lower, upper in zip(main_text_param_names_sorted, main_text_means_sorted, main_text_lower_sorted, main_text_upper_sorted):
    crosses_zero = 'crosses 0' if lower <= 0 <= upper else 'excludes 0 ***'
    print(f"{name:35s}: {mean:7.4f}  [{lower:7.4f}, {upper:7.4f}]  ({crosses_zero})")

# %% Marginal effects (mean based + rug-plot)
""" Marginal Effects of Condition on p(gain) and delta(Q) """

import numpy as np
import matplotlib.pyplot as plt

print("\n" + "="*60)
print("FINAL MARGINAL EFFECTS (MEAN-BASED)")
print("="*60)

# =========================================================
# STYLE (Journal-ready)
# =========================================================
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# =========================================================
# DATA
# =========================================================
all_trials = np.array(all_trials)
all_conditions = np.array(all_conditions)
all_p_gain = np.array(all_p_gain)
all_op_values = np.array(all_op_values)

# =========================================================
# COLORS
# =========================================================
COLORS = {
    "Approach": "#2E86AB",
    "Avoidance": "#A23B72"
}

# =========================================================
# POSTERIOR SAMPLES
# =========================================================
mu_alpha_samples = idata_unified.posterior['mu_alpha_u'].values.flatten()

mu_beta_samples = idata_unified.posterior['mu_beta_u'].values.reshape(
    -1, idata_unified.posterior['mu_beta_u'].values.shape[-1]
)

beta_cond_samples = idata_unified.posterior['beta_mod_condition'].values.reshape(
    -1, idata_unified.posterior['beta_mod_condition'].values.shape[-1]
)

beta_block_samples = idata_unified.posterior['beta_mod_block_idx'].values.reshape(
    -1, idata_unified.posterior['beta_mod_block_idx'].values.shape[-1]
)

# =========================================================
# FIXED VALUES (MEAN-BASED MARGINALIZATION)
# =========================================================
mean_wait_safe = np.nanmean(X_unified[:, 2])
mean_binary_energy = np.nanmean(X_unified[:, 3])
mean_block_idx = np.nanmean(X_unified[:, 6])

mean_p_std = 0
mean_r_std = 0
mean_op_std = 0

# =========================================================
# MARGINAL FUNCTION (FAST + CLEAN)
# =========================================================
def marginal_curve(feature_idx, interaction_idx, x_grid, x_mean, x_std, cond):
    preds = []

    for x in x_grid:
        x_std_val = (x - x_mean) / x_std

        X = np.array([
            x_std_val if feature_idx == 0 else mean_p_std,
            mean_r_std,
            mean_wait_safe,
            mean_binary_energy,
            x_std_val if feature_idx == 4 else mean_op_std,
            cond,
            mean_block_idx
        ])

        eta = mu_alpha_samples + np.sum(mu_beta_samples * X[np.newaxis, :], axis=1)

        # interactions
        eta += beta_cond_samples[:, interaction_idx] * x_std_val * cond
        eta += beta_block_samples[:, interaction_idx] * x_std_val * mean_block_idx

        p = 1 / (1 + np.exp(-eta))
        preds.append(p)

    preds = np.array(preds).T

    return (
        preds.mean(axis=0),
        np.percentile(preds, 2.5, axis=0),
        np.percentile(preds, 97.5, axis=0)
    )

# =========================================================
# RUG PLOT FUNCTION
# =========================================================
def rug_plot(ax, x, cond, color):
    y_offset = -0.02 if cond == 0 else -0.05
    ax.plot(x, np.full_like(x, y_offset), '|', color=color, alpha=0.15, markersize=8)

# =========================================================
# PLOT FUNCTION
# =========================================================
def plot_feature(ax, feature_name, x_data, x_mean, x_std, feature_idx, interaction_idx):

    x_grid = np.linspace(np.min(x_data), np.max(x_data), 150)

    for cond, label in [(0, "Approach"), (1, "Avoidance")]:
        mask = all_conditions == cond

        # --- RUG PLOT (empirical distribution only)
        rug_plot(ax, x_data[mask], cond, COLORS[label])

        # --- MODEL CURVE
        mean, lo, hi = marginal_curve(
            feature_idx,
            interaction_idx,
            x_grid,
            x_mean,
            x_std,
            cond
        )

        ax.plot(x_grid, mean, color=COLORS[label], linewidth=2.5, label=label)
        ax.fill_between(x_grid, lo, hi, color=COLORS[label], alpha=0.2)

    ax.set_xlabel(feature_name)
    ax.set_ylabel("P(foraging)")
    ax.axhline(0.5, linestyle=":", color="gray", linewidth=1)
    ax.set_ylim(-0.08, 1.0)
    ax.legend(frameon=False)

# =========================================================
# COMBINED FIGURE
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)

# --- p(gain)
plot_feature(
    axes[0],
    r'$\mathit{p}$(gain)',
    all_p_gain,
    var_mean_p_gain,
    var_std_p_gain,
    feature_idx=0,
    interaction_idx=0
)

# --- ΔQ
plot_feature(
    axes[1],
    r'$\Delta Q$-values',
    all_op_values,
    op_mean,
    op_std,
    feature_idx=4,
    interaction_idx=4
)

fig.suptitle(
    r'Marginal Effects of Condition on $\mathit{p}$(gain) and $\Delta Q$-values',
    fontsize=18,
    y=1.02
)

plt.tight_layout()
fig.savefig(path + "RESULTS/marginal_effects_final.png", dpi=300, bbox_inches='tight')

plt.show()


