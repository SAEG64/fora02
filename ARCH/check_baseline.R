# Quick script to check baseline setup in m3 model
library(readr)
library(lme4)
library(dplyr)

# Set working directory
setwd("/home/sergej/Documents/academics/dnhi/projects/AAA/FORA02/data_ana/")

# Data import
df_allSubs <- read_csv("DATA_clean/DATA_group_level/test_data.group_level_datall.csv")

# Add variables
df_allSubs['MHP_model'] <- df_allSubs['multi-heuristic policy']
df_allSubs['OP_cap'] <- df_allSubs["OP_cap"]

# Create dummy variable (same as in your main script)
df_allSubs$condition_dummy <- as.numeric(factor(df_allSubs$condition_rORp)) - 1

# Check the mapping
cat("=== CONDITION DUMMY VARIABLE MAPPING ===\n")
cat("Original condition_rORp values:", paste(unique(df_allSubs$condition_rORp), collapse=", "), "\n")
cat("Dummy variable values:", paste(unique(df_allSubs$condition_dummy), collapse=", "), "\n\n")

cat("Cross-tabulation:\n")
print(table(df_allSubs$condition_rORp, df_allSubs$condition_dummy))

cat("\nMapping verification:\n")
cat("condition_dummy = 0 represents:", unique(df_allSubs$condition_rORp[df_allSubs$condition_dummy == 0]), "\n")
cat("condition_dummy = 1 represents:", unique(df_allSubs$condition_rORp[df_allSubs$condition_dummy == 1]), "\n\n")

# Fit the m3 model
cat("=== FITTING M3 MODEL ===\n")
m3 <- glmer(
  fora_response ~ 
    (MHP_model+OP_cap)*condition_dummy +
    (1|subject_ID), 
  data=df_allSubs, family="binomial"(link = "logit"), 
  control = glmerControl(optimizer="bobyqa"))

cat("Model formula:\n")
print(formula(m3))

cat("\nFixed effects:\n")
print(fixef(m3))

cat("\nModel summary (coefficients only):\n")
coef_summary <- summary(m3)$coefficients
print(coef_summary)

cat("\n=== BASELINE INTERPRETATION ===\n")
cat("The baseline (reference) condition is: condition_dummy = 0\n")
cat("This corresponds to:", unique(df_allSubs$condition_rORp[df_allSubs$condition_dummy == 0]), "\n\n")

cat("Coefficient interpretation:\n")
cat("- (Intercept): Log-odds of foraging when OP_cap=0, MHP_model=0, and condition_dummy=0\n")
cat("- OP_cap: Effect of OP_cap in the baseline condition (condition_dummy=0)\n") 
cat("- MHP_model: Effect of MHP_model in the baseline condition (condition_dummy=0)\n")
cat("- condition_dummy: Main effect of changing from baseline (0) to comparison (1) condition\n")
cat("- OP_cap:condition_dummy: How the OP_cap effect changes when moving to condition_dummy=1\n")
cat("- MHP_model:condition_dummy: How the MHP_model effect changes when moving to condition_dummy=1\n")
