#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:33:38 2023
Plot full model recovery using PyMC
Features:
- Per-model parameter recovery (no mixing)
- Explicit alignment checks (no silent mismatch)
- Adaptive thresholds
- Safe correlation handling
- Warnings for low variance instead of silent failure

@author: sergej
"""

# %% LOAD + PLOT MODEL & PARAMETER RECOVERY
""" Load and visualize model recovery results """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os
import warnings

# Helper function to clean model names
def clean_model_name(name):
    """Remove special characters from model names for plotting"""
    return name.replace("*", "").replace("$", "").replace("\\", "")

def format_param_name(param_name):
    """Format parameter names with Greek letters, spaces, and special variables"""
    param = param_name
    
    # Remove mathit braces entirely first for p and r variables
    param = param.replace("mathit{p}", "$p$")
    param = param.replace("mathit{r}", "$r$")
    
    # Handle specific patterns with p and r variables (after removing mathit braces)
    param = param.replace("p_gain", "$p$(gain)")
    param = param.replace("r_threat", "$r$(threat)")
    
    # Replace other specific patterns (before underscore replacement)
    param = param.replace("expected_gain_naive", "expected gain")
    param = param.replace("multi_heuristic_policy", "multi-feature policy")
    param = param.replace("optimal_policy_values", "$\\Delta Q$-values")
    
    # Replace mu_ and sigma_ with descriptive names (without Greek letters)
    param = param.replace("mu_", "Group mean ")
    param = param.replace("sigma_", "Group SD ")
    
    # Replace standalone beta with Group mean beta
    if param == "beta":
        param = "Group mean $\\beta$"
    
    # Replace underscores with spaces
    param = param.replace("_", " ")
    
    # Replace alpha and beta with Greek letters
    param = param.replace(" alpha", " $\\alpha$")
    param = param.replace("alpha ", "$\\alpha$ ")
    param = param.replace(" beta", " $\\beta$")
    param = param.replace("beta ", "$\\beta$ ")
    
    # Replace remaining specific variable patterns with formatted versions (italicized)
    param = param.replace("p gain", "$p$(gain)")
    param = param.replace("r threat", "$r$(threat)")
    
    return param

def format_model_name(model_name):
    """Format model names with Greek letters and special replacements"""
    # First clean special characters
    model = model_name.replace("*", "").replace("$", "").replace("\\", "")
    
    # Replace specific patterns
    model = model.replace("mathit{p}", "$p$")
    model = model.replace("mathit{r}", "$r$")
    model = model.replace("expected gain naive", "expected gain")
    model = model.replace("multi-heuristic policy", "multi-feature policy")
    model = model.replace("optimal policy values", "$\\Delta Q$-values")
    
    # Replace mu_ and sigma_ with descriptive names (without Greek letters)
    model = model.replace("mu_", "Group mean ")
    model = model.replace("sigma_", "Group SD ")
    
    # Replace underscores with spaces
    model = model.replace("_", " ")
    
    # Replace alpha and beta with Greek letters
    model = model.replace(" alpha", " $\\alpha$")
    model = model.replace("alpha ", "$\\alpha$ ")
    model = model.replace(" beta", " $\\beta$")
    model = model.replace("beta ", "$\\beta$ ")
    
    return model

# ---------------------------------------------------------------------
# PATH
# ---------------------------------------------------------------------
path = os.path.dirname(__file__) + "/"
results_path = path + "RECOVERY_RESULTS/"

# ---------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 22,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
recovery_counts = np.load(results_path + "recovery_counts.npy")
recovery_matrix = np.load(results_path + "recovery_matrix.npy")
df = pd.read_csv(results_path + "param_recovery.csv")
mdlName = np.load(results_path + "model_names.npy", allow_pickle=True)

print("\nLoaded recovery results:")
print(f"Models: {len(mdlName)}")
print(f"Parameter samples: {len(df)}")

# ---------------------------------------------------------------------
# MODEL RECOVERY: CONFUSION MATRIX
# ---------------------------------------------------------------------
plt.figure(figsize=(12, 10))

sns.heatmap(
    recovery_matrix,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    xticklabels=[format_model_name(name) for name in mdlName],
    yticklabels=[format_model_name(name) for name in mdlName],
    cbar_kws={"label": "Recovery probability"}
)

plt.title("Model Recovery")
plt.xlabel("Recovered model")
plt.ylabel("Generating model")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(results_path + "model_recovery_confusion.png", dpi=300)
plt.show()

# ---------------------------------------------------------------------
# MODEL RECOVERY ACCURACY
# ---------------------------------------------------------------------
accuracy = np.trace(recovery_counts) / np.sum(recovery_counts)

print("\n" + "="*60)
print(f"MODEL RECOVERY ACCURACY: {accuracy:.3f}")
print("="*60)

# ---------------------------------------------------------------------
# PARAMETER DETECTION (GROUP-LEVEL ONLY)
# ---------------------------------------------------------------------

# Exclude subject-level and offset parameters only
exclude_patterns = [
    "_subj_",
    "_subject_",
    "offset"
]

true_cols = [
    c for c in df.columns
    if c.startswith("true_")
    and not any(p in c for p in exclude_patterns)
]

est_cols = [
    c.replace("true_", "est_")
    for c in true_cols
    if c.replace("true_", "est_") in df.columns
]

param_pairs = list(zip(true_cols, est_cols))

print("\nDetected GROUP-LEVEL parameters:")
for t, e in param_pairs:
    print(f"  {t}  ↔  {e}")

# ---------------------------------------------------------------------
# PARAMETER RECOVERY PLOTS (PER MODEL)
# ---------------------------------------------------------------------
for model_name in df["gen_model"].unique():

    df_m = df[df["gen_model"] == model_name]

    if len(df_m) < 5:
        continue

    clean_name = format_model_name(model_name)
    print(f"\nPlotting parameters for: {clean_name}")

    # adaptive threshold
    min_points = max(5, int(0.3 * len(df_m)))

    for true_col, est_col in param_pairs:

        if true_col not in df_m.columns or est_col not in df_m.columns:
            continue

        # -----------------------------------------------------------------
        # SAFE ALIGNMENT (NO SILENT MISMATCH)
        # -----------------------------------------------------------------
        valid_mask = (
            df_m[true_col].notna() &
            df_m[est_col].notna() &
            np.isfinite(df_m[true_col]) &
            np.isfinite(df_m[est_col])
        )

        df_valid = df_m.loc[valid_mask, [true_col, est_col]].copy()

        if len(df_valid) < min_points:
            continue

        x = df_valid[true_col].values
        y = df_valid[est_col].values

        # -----------------------------------------------------------------
        # VARIANCE CHECK (WARNING ONLY)
        # -----------------------------------------------------------------
        if np.std(x) < 1e-6:
            warnings.warn(
                f"[WARNING] Near-zero variance in TRUE parameter '{true_col}' "
                f"for model '{model_name}'. Correlation may be unreliable."
            )

        if np.std(y) < 1e-6:
            warnings.warn(
                f"[WARNING] Near-zero variance in EST parameter '{est_col}' "
                f"for model '{model_name}'."
            )

        # -----------------------------------------------------------------
        # REGRESSION
        # -----------------------------------------------------------------
        try:
            X_reg = sm.add_constant(x)
            reg_model = sm.OLS(y, X_reg).fit()

            slope = reg_model.params[1]
            intercept = reg_model.params[0]
            ci_low, ci_high = reg_model.conf_int()[1]

        except Exception as e:
            warnings.warn(f"[ERROR] Regression failed for {true_col}: {e}")
            continue

        # -----------------------------------------------------------------
        # SAFE CORRELATION
        # -----------------------------------------------------------------
        try:
            r = np.corrcoef(x, y)[0, 1]
            if np.isnan(r):
                warnings.warn(
                    f"[WARNING] Correlation is NaN for {true_col} in {model_name}"
                )
        except Exception:
            r = np.nan
            warnings.warn(
                f"[WARNING] Correlation computation failed for {true_col} in {model_name}"
            )

        # -----------------------------------------------------------------
        # PLOT
        # -----------------------------------------------------------------
        plt.figure(figsize=(6, 6))

        sns.scatterplot(x=x, y=y, alpha=0.7, edgecolor=None)

        lims = [min(x.min(), y.min()), max(x.max(), y.max())]

        # identity
        plt.plot(lims, lims, 'r--', linewidth=2, label='Identity')

        # regression
        x_vals = np.linspace(x.min(), x.max(), 100)
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, color='black', linewidth=2, label='Fit')
        
        # Calculate prediction standard error and confidence interval
        try:
            predictions = reg_model.get_prediction(sm.add_constant(x_vals))
            pred_summary = predictions.summary_frame(alpha=0.05)
            ci_lower = pred_summary['mean_ci_lower'].values
            ci_upper = pred_summary['mean_ci_upper'].values
            plt.fill_between(x_vals, ci_lower, ci_upper, alpha=0.2, color='black', label='95% CI')
        except Exception as e:
            warnings.warn(f"[WARNING] Could not compute confidence interval: {e}")

        param_name = true_col.replace("true_", "")
        formatted_param = format_param_name(param_name)

        plt.title(f"{clean_name}\n{formatted_param}")
        
        # Add statistics text box inside plot
        stats_text = f"slope = {slope:.2f}\n$r$ = {r:.2f}"
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.xlabel("True")
        plt.ylabel("Estimated")

        plt.xlim(lims)
        plt.ylim(lims)
        plt.legend()

        plt.tight_layout()

        safe_model = model_name.replace(" ", "_").replace("*", "").replace("$", "").replace("\\", "")
        safe_param = param_name.replace("beta_int_", "int_")

        fname = f"{safe_model}__{safe_param}.png"
        plt.savefig(results_path + fname, dpi=300)

        plt.show()
        # ---------------------------------------------------------------------
# SUMMARY STATISTICS (PER MODEL)
# ---------------------------------------------------------------------
print("\n" + "="*60)
print("PARAMETER RECOVERY SUMMARY (PER MODEL)")
print("="*60)

for model_name in df["gen_model"].unique():

    df_m = df[df["gen_model"] == model_name]

    if len(df_m) < 5:
        continue

    clean_name = format_model_name(model_name)
    print(f"\nModel: {clean_name}")
    print("-"*50)

    min_points = max(5, int(0.3 * len(df_m)))

    for true_col, est_col in param_pairs:

        if true_col not in df_m.columns or est_col not in df_m.columns:
            continue

        valid_mask = (
            df_m[true_col].notna() &
            df_m[est_col].notna() &
            np.isfinite(df_m[true_col]) &
            np.isfinite(df_m[est_col])
        )

        df_valid = df_m.loc[valid_mask, [true_col, est_col]].copy()

        if len(df_valid) < min_points:
            continue

        x = df_valid[true_col].values
        y = df_valid[est_col].values

        try:
            X_reg = sm.add_constant(x)
            reg_model = sm.OLS(y, X_reg).fit()

            slope = reg_model.params[1]
            ci_low, ci_high = reg_model.conf_int()[1]

            r = np.corrcoef(x, y)[0, 1]

        except Exception as e:
            warnings.warn(f"[ERROR] Summary failed for {true_col}: {e}")
            continue

        print(
            f"{true_col.replace('true_', ''):35s}: "
            f"slope = {slope:.3f} [{ci_low:.3f}, {ci_high:.3f}], "
            f"r = {r:.3f}"
        )

# %% Publication Parameter Recovery Figures
""" Publication-focused parameter recovery figures """
# =====================================================================
print("\n" + "="*60)
print("CREATING PUBLICATION PARAMETER RECOVERY FIGURES")
print("="*60)

# ---------------------------------------------------------------------
# PARAMETER LABELS
# ---------------------------------------------------------------------

pretty_param_labels = {

    # Generic
    "mu_alpha":
        r"Group mean $\mathit{\alpha}$",

    "sigma_alpha":
        r"Group SD $\mathit{\alpha}$",

    "beta":
        r"Group mean $\mathit{\beta}$",

    "sigma_beta":
        r"Group SD $\mathit{\beta}$",

    # Multi-feature means
    "beta_mathitp_gain":
        r"Group mean $\mathit{\beta}_{p(gain)}$",

    "beta_mathitr_predator":
        r"Group mean $\mathit{\beta}_{r(predator)}$",

    "beta_wait_when_safe":
        r"Group mean $\mathit{\beta}_{WWS}$",

    "beta_binary_energy_state":
        r"Group mean $\mathit{\beta}_{BES}$",

    # Multi-feature SDs
    "sigma_beta_mathitp_gain":
        r"Group SD $\mathit{\beta}_{p(gain)}$",

    "sigma_beta_mathitr_predator":
        r"Group SD $\mathit{\beta}_{r(predator)}$",

    "sigma_beta_wait_when_safe":
        r"Group SD $\mathit{\beta}_{WWS}$",

    "sigma_beta_binary_energy_state":
        r"Group SD $\mathit{\beta}_{BES}$",
}

# =====================================================================
# HELPER FUNCTION
# =====================================================================

def create_meshgrid_figure(
    models_to_plot,
    param_include_list,
    figure_title,
    save_stub
):

    # -------------------------------------------------------------
    # FILTER PARAMS
    # -------------------------------------------------------------
    param_pairs_plot = [
        (t, e)
        for (t, e) in param_pairs
        if t in param_include_list
    ]

    unique_models = [
        m for m in models_to_plot
        if m in df["gen_model"].unique()
    ]

    n_models = len(unique_models)
    n_params = len(param_pairs_plot)

    print(f"\nCreating: {figure_title}")
    print(f"{n_models} models × {n_params} parameters")

    # -------------------------------------------------------------
    # FIGURE
    # -------------------------------------------------------------
    fig, axes = plt.subplots(
        n_models,
        n_params,
        figsize=(4*n_params, 4*n_models)
    )

    # Ensure 2D
    if n_models == 1 and n_params == 1:
        axes = np.array([[axes]])
    elif n_models == 1:
        axes = axes.reshape(1, -1)
    elif n_params == 1:
        axes = axes.reshape(-1, 1)

    # -------------------------------------------------------------
    # PLOTS
    # -------------------------------------------------------------
    for model_idx, model_name in enumerate(unique_models):

        df_m = df[df["gen_model"] == model_name]
        clean_model_name_str = format_model_name(model_name)

        for param_idx, (true_col, est_col) in enumerate(param_pairs_plot):

            ax = axes[model_idx, param_idx]

            # -----------------------------------------------------
            # Skip unavailable parameters
            # -----------------------------------------------------
            if (
                true_col not in df_m.columns
                or est_col not in df_m.columns
            ):
                ax.axis('off')
                continue

            # -----------------------------------------------------
            # VALID DATA
            # -----------------------------------------------------
            valid_mask = (
                df_m[true_col].notna() &
                df_m[est_col].notna() &
                np.isfinite(df_m[true_col]) &
                np.isfinite(df_m[est_col])
            )

            df_valid = df_m.loc[
                valid_mask,
                [true_col, est_col]
            ].copy()

            if len(df_valid) < 5:
                ax.axis('off')
                continue

            x = df_valid[true_col].values
            y = df_valid[est_col].values

            if np.std(x) < 1e-6 or np.std(y) < 1e-6:
                ax.axis('off')
                continue

            # -----------------------------------------------------
            # REGRESSION
            # -----------------------------------------------------
            try:

                X_reg = sm.add_constant(x)
                reg_model_fit = sm.OLS(y, X_reg).fit()

                slope = reg_model_fit.params[1]
                intercept = reg_model_fit.params[0]

                slope_se = reg_model_fit.bse[1]

                r = np.corrcoef(x, y)[0, 1]

                # -------------------------------------------------
                # PLOT
                # -------------------------------------------------

                ax.scatter(
                    x,
                    y,
                    alpha=0.6,
                    s=20,
                    edgecolor=None
                )

                lims = [
                    min(x.min(), y.min()),
                    max(x.max(), y.max())
                ]

                # Identity
                ax.plot(
                    lims,
                    lims,
                    'r--',
                    linewidth=1,
                    alpha=0.5
                )

                # Regression line
                x_vals = np.linspace(x.min(), x.max(), 100)
                y_vals = intercept + slope * x_vals

                ax.plot(
                    x_vals,
                    y_vals,
                    'k-',
                    linewidth=1.5,
                    alpha=0.7
                )

                # -------------------------------------------------
                # CONFIDENCE INTERVAL
                # -------------------------------------------------
                predictions = reg_model_fit.get_prediction(
                    sm.add_constant(x_vals)
                )

                pred_summary = predictions.summary_frame(alpha=0.05)

                ci_lower = pred_summary['mean_ci_lower'].values
                ci_upper = pred_summary['mean_ci_upper'].values

                ax.fill_between(
                    x_vals,
                    ci_lower,
                    ci_upper,
                    alpha=0.15,
                    color='black'
                )

                # -------------------------------------------------
                # TITLE
                # -------------------------------------------------
                ax.set_title(
                    f"slope = {slope:.2f} ± {slope_se:.2f}\n"
                    f"$r$ = {r:.2f}",
                    fontsize=11,
                    pad=8
                )

            except Exception as e:

                warnings.warn(
                    f"[WARNING] Regression failed for {true_col}: {e}"
                )

            # -----------------------------------------------------
            # FORMAT
            # -----------------------------------------------------
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            ax.set_xlabel("True", fontsize=10)
            ax.set_ylabel("Est.", fontsize=10)

            ax.tick_params(labelsize=9)

            ax.grid(
                True,
                alpha=0.2,
                linestyle='--',
                linewidth=0.5
            )

            # -----------------------------------------------------
            # ROW LABELS
            # -----------------------------------------------------
            if param_idx == 0:

                ax.set_ylabel(
                    f"{clean_model_name_str}\nEst.",
                    fontsize=11
                )

            # -----------------------------------------------------
            # COLUMN LABELS
            # -----------------------------------------------------
            if model_idx == 0:

                param_name = true_col.replace("true_", "")

                formatted_param = pretty_param_labels.get(
                    param_name,
                    param_name
                )

                ax.text(
                    0.5,
                    1.20,
                    formatted_param,
                    transform=ax.transAxes,
                    ha='center',
                    va='bottom',
                    fontsize=11
                )

    # -------------------------------------------------------------
    # FINALIZE
    # -------------------------------------------------------------
    plt.suptitle(
        figure_title,
        fontsize=16,
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    png_fname = results_path + f"{save_stub}.png"
    pdf_fname = results_path + f"{save_stub}.pdf"

    plt.savefig(
        png_fname,
        dpi=300,
        bbox_inches='tight'
    )

    plt.savefig(
        pdf_fname,
        bbox_inches='tight'
    )

    print(f"Saved: {png_fname}")
    print(f"Saved: {pdf_fname}")

    plt.show()

# =====================================================================
# FIGURE 1:
# SINGLE-FEATURE MODELS
# =====================================================================

single_feature_models = [
    "* $\\mathit{p}$ gain",
    "* $\\mathit{r}$ predator",
    "marginal value",
    "optimal policy values"
]

single_feature_params = [

    "true_mu_alpha",
    "true_sigma_alpha",

    "true_beta",
    "true_sigma_beta",
]

create_meshgrid_figure(
    models_to_plot=single_feature_models,
    param_include_list=single_feature_params,
    figure_title="Parameter Recovery: Single-Feature Models",
    save_stub="parameter_recovery_single_feature"
)

# %% MULTI-FEATURE POLICY FIGURE:
""" MULTI-FEATURE POLICY — FULL GROUP-LEVEL PARAMETER RECOVERY """
# =====================================================================

print("\nCreating full MH parameter recovery meshgrid")

# ---------------------------------------------------------------------
# SELECT ALL GROUP-LEVEL PARAMETERS
# ---------------------------------------------------------------------

mh_param_pairs = []

df_m_temp = df[df["gen_model"] == "multi-heuristic policy"]

for true_col, est_col in param_pairs:

    # Exclude subject-level parameters only
    if (
        "_subj_" in true_col
        or "_subject_" in true_col
        or "offset" in true_col
    ):
        continue

    # Keep only existing columns
    if (
        true_col in df.columns
        and est_col in df.columns
    ):
        # Pre-filter: only include parameters with sufficient valid data
        valid_mask = (
            df_m_temp[true_col].notna() &
            df_m_temp[est_col].notna() &
            np.isfinite(df_m_temp[true_col]) &
            np.isfinite(df_m_temp[est_col])
        )
        
        if len(df_m_temp.loc[valid_mask]) >= 5:
            x_temp = df_m_temp.loc[valid_mask, true_col].values
            y_temp = df_m_temp.loc[valid_mask, est_col].values
            
            # Only include if variance is not near-zero
            if np.std(x_temp) >= 1e-6 and np.std(y_temp) >= 1e-6:
                mh_param_pairs.append((true_col, est_col))

print(f"Detected {len(mh_param_pairs)} group-level parameters with sufficient data")

# ---------------------------------------------------------------------
# SORT PARAMETERS
# ---------------------------------------------------------------------

desired_order = [

    # intercepts
    "true_mu_alpha",
    "true_sigma_alpha",

    # generic beta
    "true_beta",
    "true_sigma_beta",

    # MH feature means
    "true_beta_mathitp_gain",
    "true_beta_mathitr_predator",
    "true_beta_wait_when_safe",
    "true_beta_binary_energy_state",

    # MH feature SDs
    "true_sigma_beta_mathitp_gain",
    "true_sigma_beta_mathitr_predator",
    "true_sigma_beta_wait_when_safe",
    "true_sigma_beta_binary_energy_state",
]

mh_param_pairs_sorted = []

# First: desired order (only those in mh_param_pairs)
for desired in desired_order:

    for t, e in mh_param_pairs:

        if t == desired:
            mh_param_pairs_sorted.append((t, e))

# Second: append remaining parameters automatically
for t, e in mh_param_pairs:

    if (t, e) not in mh_param_pairs_sorted:
        mh_param_pairs_sorted.append((t, e))

mh_param_pairs = mh_param_pairs_sorted

# Now the grid will only contain parameters with sufficient data
n_cols = 4
n_params = len(mh_param_pairs)
n_rows = int(np.ceil(n_params / n_cols))

print(f"Creating {n_rows} rows × {n_cols} columns grid for {n_params} parameters")

# Create the figure and axes
fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(4*n_cols, 4*n_rows)
)

axes = np.array(axes).reshape(n_rows, n_cols)

# Get data for multi-heuristic policy model
df_m = df[df["gen_model"] == "multi-heuristic policy"]

# ---------------------------------------------------------------------
# PARAMETER LABEL FUNCTION
# ---------------------------------------------------------------------

def format_mh_param_label(param_name):

    # -------------------------------------------------------------
    # INTERCEPTS
    # -------------------------------------------------------------
    if param_name == "mu_alpha":
        return r"Group mean $\mathit{\alpha}$"

    if param_name == "sigma_alpha":
        return r"Group SD $\mathit{\alpha}$"

    # -------------------------------------------------------------
    # GENERIC BETAS
    # -------------------------------------------------------------
    if param_name == "beta":
        return r"Group mean $\mathit{\beta}$"

    if param_name == "sigma_beta":
        return r"Group SD $\mathit{\beta}$"

    # -------------------------------------------------------------
    # DETERMINE PREFIX
    # -------------------------------------------------------------
    if param_name.startswith("beta_"):

        prefix = r"Group mean $\mathit{\beta}$: "
        remainder = param_name[len("beta_"):]

    elif param_name.startswith("sigma_beta_"):

        prefix = r"Group SD $\mathit{\beta}$: "
        remainder = param_name[len("sigma_beta_"):]

    else:
        return param_name.replace("_", " ")

    # -------------------------------------------------------------
    # REMOVE LEADING UNDERSCORES
    # -------------------------------------------------------------
    remainder = remainder.strip("_")

    # -------------------------------------------------------------
    # CLEAN VARIABLE NAMES
    # -------------------------------------------------------------

    # p(gain)
    remainder = remainder.replace(
        "mathitp_gain",
        r"$\mathit{p}$(gain)"
    )

    # r(threat)
    remainder = remainder.replace(
        "mathitr_predator",
        r"$\mathit{r}$(threat)"
    )

    # WWS
    remainder = remainder.replace(
        "wait_when_safe",
        "WWS"
    )

    # BES
    remainder = remainder.replace(
        "binary_energy_state",
        "BES"
    )

    # -------------------------------------------------------------
    # INTERACTIONS
    # -------------------------------------------------------------

    # remove int markers
    remainder = remainder.replace("beta_int_", "")
    remainder = remainder.replace("int_", "")

    # interaction separator
    remainder = remainder.replace("_x_", " × ")

    # remove remaining underscores
    remainder = remainder.replace("_", " ")

    return prefix + remainder

# ---------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------

for idx, (true_col, est_col) in enumerate(mh_param_pairs):

    row = idx // n_cols
    col = idx % n_cols

    ax = axes[row, col]

    # -------------------------------------------------------------
    # VALID DATA
    # -------------------------------------------------------------
    valid_mask = (
        df_m[true_col].notna() &
        df_m[est_col].notna() &
        np.isfinite(df_m[true_col]) &
        np.isfinite(df_m[est_col])
    )

    df_valid = df_m.loc[
        valid_mask,
        [true_col, est_col]
    ].copy()

    if len(df_valid) < 5:

        ax.axis('off')
        continue

    x = df_valid[true_col].values
    y = df_valid[est_col].values

    if np.std(x) < 1e-6 or np.std(y) < 1e-6:

        ax.axis('off')
        continue

    # -------------------------------------------------------------
    # REGRESSION
    # -------------------------------------------------------------
    try:

        X_reg = sm.add_constant(x)
        reg_model_fit = sm.OLS(y, X_reg).fit()

        slope = reg_model_fit.params[1]
        intercept = reg_model_fit.params[0]

        slope_se = reg_model_fit.bse[1]

        r = np.corrcoef(x, y)[0, 1]

        # ---------------------------------------------------------
        # SCATTER
        # ---------------------------------------------------------
        ax.scatter(
            x,
            y,
            alpha=0.6,
            s=22,
            edgecolor=None
        )

        lims = [
            min(x.min(), y.min()),
            max(x.max(), y.max())
        ]

        # identity
        ax.plot(
            lims,
            lims,
            'r--',
            linewidth=1,
            alpha=0.5
        )

        # regression
        x_vals = np.linspace(x.min(), x.max(), 100)
        y_vals = intercept + slope * x_vals

        ax.plot(
            x_vals,
            y_vals,
            'k-',
            linewidth=1.5,
            alpha=0.7
        )

        # ---------------------------------------------------------
        # CONFIDENCE BAND
        # ---------------------------------------------------------
        predictions = reg_model_fit.get_prediction(
            sm.add_constant(x_vals)
        )

        pred_summary = predictions.summary_frame(alpha=0.05)

        ci_lower = pred_summary['mean_ci_lower'].values
        ci_upper = pred_summary['mean_ci_upper'].values

        ax.fill_between(
            x_vals,
            ci_lower,
            ci_upper,
            alpha=0.15,
            color='black'
        )

        # ---------------------------------------------------------
        # TITLE
        # ---------------------------------------------------------
        ax.set_title(
            f"slope = {slope:.2f} ± {slope_se:.2f}\n"
            f"$r$ = {r:.2f}",
            fontsize=11,
            pad=8
        )

    except Exception as e:

        warnings.warn(
            f"[WARNING] Regression failed for {true_col}: {e}"
        )

    # -------------------------------------------------------------
    # PARAMETER LABEL
    # -------------------------------------------------------------
    param_name = true_col.replace("true_", "")
    formatted_param = format_mh_param_label(param_name)

    ax.text(
        0.5,
        1.18,
        formatted_param,
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=11
    )

    # -------------------------------------------------------------
    # FORMAT
    # -------------------------------------------------------------
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("True", fontsize=10)
    ax.set_ylabel("Est.", fontsize=10)

    ax.tick_params(labelsize=9)

    ax.grid(
        True,
        alpha=0.2,
        linestyle='--',
        linewidth=0.5
    )

# ---------------------------------------------------------------------
# REMOVE EMPTY PANELS
# ---------------------------------------------------------------------

for idx in range(len(mh_param_pairs), n_rows * n_cols):

    row = idx // n_cols
    col = idx % n_cols

    axes[row, col].axis('off')

# ---------------------------------------------------------------------
# FINALIZE
# ---------------------------------------------------------------------

plt.suptitle(
    "Parameter Recovery: Multi-Feature Policy",
    fontsize=16,
    y=0.995
)

plt.tight_layout(rect=[0, 0, 1, 0.985])

png_fname = results_path + "parameter_recovery_multifeature_all.png"
pdf_fname = results_path + "parameter_recovery_multifeature_all.pdf"

plt.savefig(
    png_fname,
    dpi=300,
    bbox_inches='tight'
)

plt.savefig(
    pdf_fname,
    bbox_inches='tight'
)

print(f"Saved: {png_fname}")
print(f"Saved: {pdf_fname}")

plt.show()

