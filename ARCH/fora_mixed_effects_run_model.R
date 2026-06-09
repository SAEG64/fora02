# ==============================================================================
# Comparing logistic stats models (mixed effects) of policy interaction with 
# forest condition (low and high threat containing forest; fixed effects) 
# when controlling for subject variance (random effects)
# ==============================================================================
# Packages
library(readr)
library(lme4)
library(ggeffects)
library(splines)
library(ggplot2)
library(dplyr)
# Get the directory of the script (works outside RStudio too)
filepath <- "/home/sergej/Documents/academics/dnhi/projects/AAA/FORA02/data_ana/"

# Set working directory
setwd(filepath)
# Data import
df_allSubs <- read_csv("DATA_clean/DATA_group_level/test_data.group_level_datall.csv")
# Add WWS model and bin_e
df_allSubs['bin_e_state'] <- df_allSubs['** binary energy state']
df_allSubs['wait_when_safe'] <- df_allSubs['** wait when safe']
df_allSubs['OP_values'] <- df_allSubs['optimal policy values']
df_allSubs['MHP_model'] <- df_allSubs['multi-heuristic policy']
df_allSubs['OP_cap'] <- df_allSubs["OP_cap"]
df_allSubs['conditions'] <- factor(df_allSubs$condition_rORp)
# Filter data
subset <- filter(df_allSubs, conditions == 1)
# 
# # Run hierarchical models for comparing info criterion in avoidance forests
# mdl <- glmer(
#   fora_response ~ 
#     OP_cap +
#     (1|subject_ID), 
#   data=subset, family="binomial"(link = "logit"), 
#   control = glmerControl(optimizer="bobyqa"))
# summary(m_OP)
# # BIC = 5510.9
# 
# mdl <- glmer(
#   fora_response ~ 
#     MHP_model +
#     (1|subject_ID), 
#   data=subset, family="binomial"(link = "logit"), 
#   control = glmerControl(optimizer="bobyqa"))
# summary(MHP_model)
# # BIC = 5250.1

# Run mixed effects models
# First check the original data and factor levels
cat("\n=== ORIGINAL DATA ===\n")
cat("Original unique values in condition_rORp:", paste(unique(df_allSubs$condition_rORp), collapse=", "), "\n")

# Properly set reference level
cat("\n=== SETTING REFERENCE LEVEL ===\n")
# First ensure it's a factor, then relevel it, then save it back to the dataframe
# Create dummy variable for condition (1 = high threat, 0 = low threat)
df_allSubs$condition_dummy <- as.numeric(factor(df_allSubs$condition_rORp))
cat("Original unique values in condition_rORp:", paste(unique(df_allSubs$condition_rORp), collapse=", "), "\n")
# Create dummy variable with explicit baseline reference
df_allSubs$condition_dummy <- 1 - (as.numeric(factor(df_allSubs$condition_rORp)) - 1)
cat("Dummy variable values (0 = high threat, 1 = low threat):", paste(unique(df_allSubs$condition_dummy), collapse=", "), "\n")
cat("Baseline reference level: 0 (high threat condition)\n")
cat("Comparison level: 1 (low threat condition)\n")

mdl <- glmer(
  fora_response ~ 
    OP_cap*condition_dummy +
    (1|subject_ID), 
  data=df_allSubs, family="binomial"(link = "logit"), 
  control = glmerControl(optimizer="bobyqa"))
ss <- getME(mdl,c("theta","fixef"))
#ss <- getME(mdl,"ALL")
m1 <- update(
  mdl,start=ss,control=glmerControl(
    optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
summary(m1)

mdl <- glmer(
  fora_response ~ 
    MHP_model*condition_dummy +
    (1|subject_ID), 
  data=df_allSubs, family="binomial"(link = "logit"), 
  control = glmerControl(optimizer="bobyqa"))
ss <- getME(mdl,c("theta","fixef"))
#ss <- getME(mdl,"ALL")
m2 <- update(
  mdl,start=ss,control=glmerControl(
    optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
summary(m2)

mdl <- glmer(
  fora_response ~ 
    (MHP_model+OP_cap)*condition_dummy +
    (1|subject_ID), 
  data=df_allSubs, family="binomial"(link = "logit"), 
  control = glmerControl(optimizer="bobyqa"))
ss <- getME(mdl,c("theta","fixef"))
#ss <- getME(mdl,"ALL")
m3 <- update(
  mdl,start=ss,control=glmerControl(
    optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
summary(m3)

# ==============================================================================
# CHECKING BASELINE SETUP IN M3 MODEL
# ==============================================================================
cat("\n=== CHECKING BASELINE SETUP IN M3 MODEL ===\n")
cat("Dummy variable mapping:\n")
cat("condition_dummy = 0 represents:", unique(df_allSubs$condition_rORp[df_allSubs$condition_dummy == 0]), "\n")
cat("condition_dummy = 1 represents:", unique(df_allSubs$condition_rORp[df_allSubs$condition_dummy == 1]), "\n\n")

cat("Model coefficients interpretation:\n")
cat("(Intercept): Effect when all predictors = 0 (i.e., OP_cap=0, MHP_model=0, condition_dummy=0)\n")
cat("condition_dummy: Main effect of moving from baseline (0) to comparison (1) condition\n")
cat("OP_cap:condition_dummy: How much the OP_cap effect changes when moving from baseline to comparison\n")
cat("MHP_model:condition_dummy: How much the MHP_model effect changes when moving from baseline to comparison\n\n")

cat("Fixed effects from m3 model:\n")
print(fixef(m3))

cat("\nModel formula:\n")
print(formula(m3))

cat("\nData summary for condition variables:\n")
table(df_allSubs$condition_rORp, df_allSubs$condition_dummy)

# ==============================================================================
# UNDERSTANDING WHY INTERACTIONS MIGHT BE SIGNIFICANT IN COMBINED MODEL
# BUT NOT IN SEPARATE MODELS
# ==============================================================================

# Load required packages
if (!require(car)) install.packages("car")
library(car)

# First, check correlation between predictors
cat("\n=== CORRELATION BETWEEN PREDICTORS ===\n")
cor_test <- cor.test(df_allSubs$MHP_model, df_allSubs$OP_cap)
print(cor_test)
cat("Correlation coefficient: ", cor_test$estimate, "\n")

# Run Type III tests on all models for proper comparison
cat("\n=== TYPE III TESTS FOR ALL MODELS ===\n")

# Model 1 (OP_cap interaction only)
cat("\nModel 1 (OP_cap × condition):\n")
cat("--------------------------------\n")
Anova(m1, type="III")

# Model 2 (MHP_model interaction only)
cat("\nModel 2 (MHP_model × condition):\n")
cat("--------------------------------\n")
Anova(m2, type="III")

# Model 3 (both interactions)
cat("\nModel 3 (both interactions):\n")
cat("--------------------------------\n")
Anova(m3, type="III")

# Compare models using AIC and BIC
cat("\n=== MODEL COMPARISON ===\n")
models_comparison <- data.frame(
  Model = c("m1 (OP_cap × condition)", 
            "m2 (MHP_model × condition)", 
            "m3 (both interactions)"),
  AIC = c(AIC(m1), AIC(m2), AIC(m3)),
  BIC = c(BIC(m1), BIC(m2), BIC(m3)),
  LogLik = c(logLik(m1), logLik(m2), logLik(m3))
)
print(models_comparison)

# Create models without interactions for LRT tests
m1_no_interaction <- glmer(
  fora_response ~ OP_cap + condition_dummy + (1|subject_ID),
  data=df_allSubs, family="binomial"(link="logit"),
  control=glmerControl(optimizer="bobyqa"))

m2_no_interaction <- glmer(
  fora_response ~ MHP_model + condition_dummy + (1|subject_ID),
  data=df_allSubs, family="binomial"(link="logit"),
  control=glmerControl(optimizer="bobyqa"))

m3_no_interaction <- glmer(
  fora_response ~ MHP_model + OP_cap + condition_dummy + (1|subject_ID),
  data=df_allSubs, family="binomial"(link="logit"),
  control=glmerControl(optimizer="bobyqa"))

# Likelihood Ratio Tests for interactions
cat("\n=== LIKELIHOOD RATIO TESTS FOR INTERACTIONS ===\n")
cat("\nModel 1 vs No Interaction:\n")
print(anova(m1_no_interaction, m1))

cat("\nModel 2 vs No Interaction:\n")
print(anova(m2_no_interaction, m2))

cat("\nModel 3 vs No Interaction:\n")
print(anova(m3_no_interaction, m3))

# Check for suppression effects
cat("\n=== INVESTIGATING SUPPRESSION EFFECTS ===\n")
cat("Comparing interaction coefficients across models:\n\n")

# Extract coefficients safely
# First check what coefficients are available
cat("Available coefficients in m1:\n")
print(names(fixef(m1)))
cat("\nAvailable coefficients in m2:\n")
print(names(fixef(m2)))
cat("\nAvailable coefficients in m3:\n")
print(names(fixef(m3)))

# Extract coefficients with proper names
coef_names_m1 <- names(fixef(m1))
coef_names_m2 <- names(fixef(m2))
coef_names_m3 <- names(fixef(m3))

# Find interaction terms
op_inter_name <- coef_names_m1[grep("OP_cap.*condition_dummy", coef_names_m1)]
mhp_inter_name <- coef_names_m2[grep("MHP_model.*condition_dummy", coef_names_m2)]
op_inter_name_m3 <- coef_names_m3[grep("OP_cap.*condition_dummy", coef_names_m3)]
mhp_inter_name_m3 <- coef_names_m3[grep("MHP_model.*condition_dummy", coef_names_m3)]

# Extract coefficients safely
coef_m1 <- if(length(op_inter_name) > 0) fixef(m1)[op_inter_name] else NA
coef_m2 <- if(length(mhp_inter_name) > 0) fixef(m2)[mhp_inter_name] else NA
coef_m3_op <- if(length(op_inter_name_m3) > 0) fixef(m3)[op_inter_name_m3] else NA
coef_m3_mhp <- if(length(mhp_inter_name_m3) > 0) fixef(m3)[mhp_inter_name_m3] else NA

# Create comparison table
coef_comparison <- data.frame(
  Interaction = c("OP_cap:condition_dummy", "MHP_model:condition_dummy"),
  Separate_Model = c(coef_m1, coef_m2),
  Combined_Model = c(coef_m3_op, coef_m3_mhp),
  Percent_Change = c(
    ifelse(!is.na(coef_m1) && !is.na(coef_m3_op) && abs(coef_m1) > 0, 
           (coef_m3_op - coef_m1)/abs(coef_m1)*100, NA),
    ifelse(!is.na(coef_m2) && !is.na(coef_m3_mhp) && abs(coef_m2) > 0, 
           (coef_m3_mhp - coef_m2)/abs(coef_m2)*100, NA)
  )
)
print(coef_comparison)

# ==============================================================================
# EXPLANATION OF FINDINGS
# ==============================================================================
cat("\n=== EXPLANATION OF FINDINGS ===\n")
cat("When interactions are significant in the combined model but not separately:\n\n")
cat("1. MULTICOLLINEARITY: The correlation between predictors can mask individual effects\n")
cat("   - MHP_model and OP_cap correlation: ", round(cor_test$estimate, 4), "\n\n")

cat("2. SUPPRESSION EFFECTS: Each predictor may suppress irrelevant variance in the other\n")
cat("   - Including both predictors can 'clean up' their respective interaction effects\n\n")

cat("3. MODEL SPECIFICATION: If both interactions truly exist, separate models are misspecified\n")
cat("   - Omitted variable bias can underestimate interaction effects\n\n")

cat("4. CONDITIONAL SIGNIFICANCE: The effect of one interaction may depend on controlling for the other\n")
cat("   - Only the combined model properly accounts for the covariance structure\n\n")

cat("5. PRACTICAL IMPLICATION: If the combined model (m3) shows better fit (lower AIC/BIC)\n")
cat("   and significant interactions, it's likely the more appropriate model to interpret\n")

# ==============================================================================
# ADDRESSING THE SPECIFIC RESEARCH QUESTION
# ==============================================================================
cat("\n=== RESEARCH QUESTION: DO PARTICIPANTS BECOME MORE OPTIMAL COMPARED TO MHP MODEL? ===\n")

# Extract relevant information from Model 3 (the combined model)
mhp_coef <- fixef(m3)["MHP_model"]
mhp_inter_coef <- fixef(m3)["MHP_model:condition_dummy"]
op_coef <- fixef(m3)["OP_cap"]
op_inter_coef <- fixef(m3)["OP_cap:condition_dummy"]

# Extract p-values from Anova Type III test
anova_m3 <- Anova(m3, type="III")
mhp_inter_p <- anova_m3["MHP_model:condition_dummy", "Pr(>Chisq)"]
op_inter_p <- anova_m3["OP_cap:condition_dummy", "Pr(>Chisq)"]

cat("\n1. MHP_model × condition interaction\n")
cat("   Coefficient:", round(mhp_inter_coef, 4), "\n")
cat("   p-value:", round(mhp_inter_p, 4), "\n")
cat("   Significance:", if(mhp_inter_p < 0.05) "Significant" else "Not significant", "\n\n")

cat("2. OP_cap × condition interaction\n")
cat("   Coefficient:", round(op_inter_coef, 4), "\n")
cat("   p-value:", round(op_inter_p, 4), "\n")
cat("   Significance:", if(op_inter_p < 0.05) "Significant" else "Not significant", "\n\n")

cat("3. Interpretation for MHP_model:\n")
if(mhp_inter_p < 0.05) {
  if(mhp_inter_coef > 0) {
    cat("   The effect of MHP_model on foraging behavior is STRONGER in high threat conditions (condition_dummy=1)\n")
    cat("   This suggests participants rely MORE on MHP-type strategies in high threat forests\n\n")
  } else {
    cat("   The effect of MHP_model on foraging behavior is WEAKER in high threat conditions (condition_dummy=1)\n")
    cat("   This suggests participants rely LESS on MHP-type strategies in high threat forests\n\n")
  }
} else {
  cat("   There is no significant difference in how MHP_model affects behavior across conditions\n\n")
}

cat("4. Interpretation for OP_cap:\n")
if(op_inter_p < 0.05) {
  if(op_inter_coef > 0) {
    cat("   The effect of OP_cap on foraging behavior is STRONGER in high threat conditions (condition_dummy=1)\n")
    cat("   This suggests participants become MORE optimal (per OP_cap) in high threat forests\n\n")
  } else {
    cat("   The effect of OP_cap on foraging behavior is WEAKER in high threat conditions (condition_dummy=1)\n")
    cat("   This suggests participants become LESS optimal (per OP_cap) in high threat forests\n\n")
  }
} else {
  cat("   There is no significant difference in how OP_cap affects behavior across conditions\n\n")
}

cat("5. Comparing MHP_model vs OP_cap:\n")
if(mhp_inter_p < 0.05 || op_inter_p < 0.05) {
  cat("   Comparing the coefficient magnitudes and directions:\n")
  cat("   - MHP_model coefficient change:", round(mhp_inter_coef, 4), "\n")
  cat("   - OP_cap coefficient change:", round(op_inter_coef, 4), "\n\n")
  
  if(abs(op_inter_coef) > abs(mhp_inter_coef) && op_inter_coef > 0) {
    cat("   The OP_cap interaction is stronger and positive, suggesting participants\n")
    cat("   become MORE optimal relative to the MHP model in high threat conditions\n\n")
  } else if(abs(mhp_inter_coef) > abs(op_inter_coef) && mhp_inter_coef > 0) {
    cat("   The MHP_model interaction is stronger and positive, suggesting participants\n")
    cat("   follow the MHP model MORE than becoming optimal in high threat conditions\n\n")
  } else if(op_inter_coef > 0 && mhp_inter_coef < 0) {
    cat("   OP_cap effect increases while MHP_model effect decreases in high threat conditions,\n")
    cat("   strongly suggesting participants become MORE optimal relative to the MHP model\n\n")
  } else if(op_inter_coef < 0 && mhp_inter_coef > 0) {
    cat("   MHP_model effect increases while OP_cap effect decreases in high threat conditions,\n")
    cat("   suggesting participants follow the MHP model MORE than becoming optimal\n\n")
  } else {
    cat("   Both effects change in the same direction, requiring careful interpretation of magnitudes\n\n")
  }
} else {
  cat("   Neither interaction is significant, suggesting the relationship between behavior\n")
  cat("   and both models (MHP and OP_cap) is consistent across conditions\n\n")
}

cat("6. CONCLUSION:\n")
if((op_inter_p < 0.05 && op_inter_coef > 0) && (mhp_inter_p >= 0.05 || mhp_inter_coef < 0)) {
  cat("   Evidence suggests participants DO become more optimal compared to the MHP model\n")
  cat("   in high threat conditions\n\n")
} else if((mhp_inter_p < 0.05 && mhp_inter_coef > 0) && (op_inter_p >= 0.05 || op_inter_coef < 0)) {
  cat("   Evidence suggests participants follow the MHP model MORE than becoming optimal\n") 
  cat("   in high threat conditions\n\n")
} else if(op_inter_p < 0.05 && mhp_inter_p < 0.05) {
  if(op_inter_coef > mhp_inter_coef) {
    cat("   The OP_cap interaction effect is stronger than the MHP_model interaction,\n")
    cat("   suggesting participants DO become more optimal compared to the MHP model\n\n")
  } else {
    cat("   The MHP_model interaction effect is stronger than the OP_cap interaction,\n")
    cat("   suggesting participants follow the MHP model MORE than becoming optimal\n\n")
  }
} else {
  cat("   No significant evidence that participants change their strategy across conditions\n\n")
}

# ==============================================================================
# VISUALIZING THE MODEL COMPARISONS
# ==============================================================================

# Set publication-ready theme and parameters
library(gridExtra)
library(grid)

# Create publication theme with larger text
pub_theme <- theme_bw() +
  theme(
    text = element_text(size = 18, family = "sans"),
    plot.title = element_text(size = 20, face = "bold", hjust = 0),
    axis.title = element_text(size = 18, face = "bold"),
    axis.text = element_text(size = 16),
    legend.title = element_text(size = 18, face = "bold"),
    legend.text = element_text(size = 16),
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    strip.text = element_text(size = 16, face = "bold"),
    strip.background = element_rect(fill = "white", color = "black")
  )

# Create custom condition labels
condition_labels <- c("0" = "Approach", "1" = "Avoidance")

# Create visualization to compare interaction effects
op_pred <- ggpredict(m3, terms = c("OP_cap [minmax]", "condition_dummy"))
mhp_pred <- ggpredict(m3, terms = c("MHP_model [minmax]", "condition_dummy"))

# Plot 1: OP_cap effect by condition
plot_op <- ggplot(op_pred, aes(x = x, y = predicted, color = group)) +
  geom_line(size = 1.5) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, fill = group), alpha = 0.2) +
  labs(
    title = expression(bold(italic("OP") ~ "values + cap Effect by Condition")),
    x = expression(bold(italic("OP") ~ "values + cap")), 
    y = "Predicted Probability of Foraging",
    color = "Condition",
    fill = "Condition"
  ) +
  scale_color_discrete(labels = condition_labels) +
  scale_fill_discrete(labels = condition_labels) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  pub_theme

# Plot 2: MHP effect by condition
plot_mhp <- ggplot(mhp_pred, aes(x = x, y = predicted, color = group)) +
  geom_line(size = 1.5) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, fill = group), alpha = 0.2) +
  labs(
    title = "Multi-heuristic Policy Effect by Condition",
    x = "Multi-heuristic policy", 
    y = "Predicted Probability of Foraging",
    color = "Condition",
    fill = "Condition"
  ) +
  scale_color_discrete(labels = condition_labels) +
  scale_fill_discrete(labels = condition_labels) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  pub_theme

# Calculate and plot model-predicted probabilities over the range of both predictors
# This helps visualize which model better explains behavior in each condition
# Create predictions for the empirical data
df_allSubs$predicted <- predict(m3, type = "response")

# Create interaction plots using the empirical data
# Plot 3: OP_cap predictions with empirical data
plot_op_empirical <- ggplot(df_allSubs, aes(x = OP_cap, y = predicted, color = factor(condition_dummy))) +
  geom_point(alpha = 0.4, size = 1.2) +
  geom_smooth(method = "loess", se = TRUE, size = 1.5, span = 0.7) +
  labs(
    title = expression(bold(italic("OP") ~ "values + cap Effect (Empirical Data)")),
    x = expression(bold(italic("OP") ~ "values + cap")), 
    y = "Predicted Probability of Foraging",
    color = "Condition"
  ) +
  scale_color_discrete(labels = condition_labels) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  pub_theme

# Plot 4: MHP predictions with empirical data
plot_mhp_empirical <- ggplot(df_allSubs, aes(x = MHP_model, y = predicted, color = factor(condition_dummy))) +
  geom_point(alpha = 0.4, size = 1.2) +
  geom_smooth(method = "loess", se = TRUE, size = 1.5, span = 0.7) +
  labs(
    title = "Multi-heuristic Policy Effect (Empirical Data)",
    x = "Multi-heuristic policy", 
    y = "Predicted Probability of Foraging",
    color = "Condition"
  ) +
  scale_color_discrete(labels = condition_labels) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  pub_theme


# Create prediction grid for heatmaps
pred_grid <- expand.grid(
  OP_cap = seq(min(df_allSubs$OP_cap, na.rm = TRUE), 
               max(df_allSubs$OP_cap, na.rm = TRUE), length.out = 50),
  MHP_model = seq(min(df_allSubs$MHP_model, na.rm = TRUE), 
                  max(df_allSubs$MHP_model, na.rm = TRUE), length.out = 50),
  condition_dummy = c(0, 1),
  subject_ID = df_allSubs$subject_ID[1]  # Use first subject for prediction
)

# Generate predictions
pred_grid$predicted <- predict(m3, newdata = pred_grid, type = "response", allow.new.levels = TRUE)

# Create separate datasets for each condition
low_threat <- subset(pred_grid, condition_dummy == 1)  # condition_dummy = 1 is "Avoidance"
high_threat <- subset(pred_grid, condition_dummy == 0)  # condition_dummy = 0 is "Approach"

# Plot 5: Heatmap for Approach condition
plot_approach <- ggplot(high_threat, aes(x = OP_cap, y = MHP_model, fill = predicted)) +
  geom_tile() +
  scale_fill_gradient2(
    low = "#2166AC", mid = "#F7F7F7", high = "#B2182B", 
    midpoint = 0.5, name = "P(Forage)",
    breaks = seq(0, 1, 0.2),
    labels = seq(0, 1, 0.2)
  ) +
  labs(
    title = "Approach Condition",
    x = expression(bold(italic("OP") ~ "values + cap")),
    y = "Multi-heuristic policy"
  ) +
  pub_theme +
  theme(legend.position = "right")

# Plot 6: Heatmap for Avoidance condition
plot_avoidance <- ggplot(low_threat, aes(x = OP_cap, y = MHP_model, fill = predicted)) +
  geom_tile() +
  scale_fill_gradient2(
    low = "#2166AC", mid = "#F7F7F7", high = "#B2182B", 
    midpoint = 0.5, name = "P(Forage)",
    breaks = seq(0, 1, 0.2),
    labels = seq(0, 1, 0.2)
  ) +
  labs(
    title = "Avoidance Condition",
    x = expression(bold(italic("OP") ~ "values + cap")),
    y = "Multi-heuristic policy"
  ) +
  pub_theme +
  theme(legend.position = "right")

# Create comprehensive publication-ready combined plot
combined_plot <- grid.arrange(
  plot_op, plot_mhp,
  plot_op_empirical, plot_mhp_empirical,
  plot_approach, plot_avoidance,
  ncol = 2, nrow = 3,
  top = textGrob("Mixed Effects Model: Foraging Behavior Across Conditions and Predictors", 
                 gp = gpar(fontsize = 22, fontface = "bold"))
)

# Export high-quality plots for publication
ggsave("mixed_effects_foraging_analysis.png", combined_plot, 
       width = 14, height = 18, dpi = 300, bg = "white")

ggsave("mixed_effects_foraging_analysis.pdf", combined_plot, 
       width = 14, height = 18, bg = "white")

# Also create individual plots for potential separate use
ggsave("interaction_plots_combined.png", 
       grid.arrange(plot_op, plot_mhp, ncol = 2,
                   top = textGrob("Model Interaction Effects", 
                                 gp = gpar(fontsize = 20, fontface = "bold"))), 
       width = 12, height = 6, dpi = 300, bg = "white")

ggsave("empirical_plots_combined.png", 
       grid.arrange(plot_op_empirical, plot_mhp_empirical, ncol = 2,
                   top = textGrob("Empirical Data with Model Predictions", 
                                 gp = gpar(fontsize = 20, fontface = "bold"))), 
       width = 12, height = 6, dpi = 300, bg = "white")

ggsave("heatmaps_combined.png", 
       grid.arrange(plot_approach, plot_avoidance, ncol = 2,
                   top = textGrob("Joint Effects Heatmaps", 
                                 gp = gpar(fontsize = 20, fontface = "bold"))), 
       width = 12, height = 6, dpi = 300, bg = "white")

# Print a clear interpretation based on visualizations
cat("\n=== VISUAL INTERPRETATION ===\n")
cat("The plots above show how each model predicts foraging behavior across conditions.\n\n")
cat("If the", expression(paste(italic("OP"), " values + cap")), "lines diverge more between conditions than the multi-heuristic policy lines,\n")
cat("this suggests participants become more optimal relative to the multi-heuristic policy model.\n\n")
cat("The heatmaps show the joint effects of both predictors in each condition.\n")
cat("Areas where the color gradient changes differently across conditions indicate\n")
cat("where participant behavior shifts between models.\n\n")
cat("Publication-ready plots exported:\n")
cat("- mixed_effects_foraging_analysis.png/.pdf (combined plot)\n")
cat("- interaction_plots_combined.png (interaction effects only)\n")
cat("- empirical_plots_combined.png (empirical data with predictions)\n")
cat("- heatmaps_combined.png (joint effects heatmaps)\n")
