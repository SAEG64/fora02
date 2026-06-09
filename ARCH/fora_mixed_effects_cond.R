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
filepath <- paste(dirname(rstudioapi::getSourceEditorContext()$path), "/", sep = "")

# Set working directory
setwd(filepath)
# Data import
df_approach <- read_csv("DATA_clean/DATA_group_level/test_data.group_level_approach.csv")
df_avoidanc <- read_csv("DATA_clean/DATA_group_level/test_data.group_level_avoidance.csv")
# Rename variables approach forests
df_approach['ternary'] <- df_approach['ternary state']
df_approach['p_succ'] <- df_approach['** $\\mathit{p}$ success']
df_approach['OP_values'] <- df_approach['optimal policy values']
df_approach['condition'] <- df_approach['condition_rORp']
df_approach['multiheuristic'] <- df_approach['multi-heuristic policy']
# Rename variables avoidance forests
df_avoidanc['ternary'] <- df_avoidanc['ternary state']
df_avoidanc['p_succ'] <- df_avoidanc['** $\\mathit{p}$ success']
df_avoidanc['OP_values'] <- df_avoidanc['optimal policy values']
df_avoidanc['condition'] <- df_avoidanc['condition_rORp']
df_avoidanc['multiheuristic'] <- df_avoidanc['multi-heuristic policy']

##### Approach forests
# Interaction for multivariate p_succ
mod.appro.pSucc <- glmer(fora_response ~ factor(ternary) +
                    p_succ +
                    (1 | subject_ID),
                  data=df_approach,
                  family="binomial"(link = "logit"))
# isSingular(mod.appro.pSucc)
summary(mod.appro.pSucc)

# Interaction for multivariate OP
mod.appro.OP <- glmer(fora_response ~ factor(ternary) +
                          OP_values +
                          (1 | subject_ID),
                        data=df_approach,
                        family="binomial"(link = "logit"))
# isSingular(mod.appro.OP)
summary(mod.appro.OP)

##### Avoidance forests
# Interaction for multivariate p_succ
mod.avoid.pSucc <- glmer(fora_response ~ factor(ternary) +
                           p_succ +
                           (1 | subject_ID),
                         data=df_avoidanc,
                         family="binomial"(link = "logit"))
# isSingular(mod.avoid.pSucc)
summary(mod.avoid.pSucc)

# Interaction for multivariate OP
mod.avoid.OP <- glmer(fora_response ~ factor(ternary) +
                        OP_values +
                        (1 | subject_ID),
                      data=df_avoidanc,
                      family="binomial"(link = "logit"))
# isSingular(mod.avoid.OP)
summary(mod.avoid.OP)


print('Model comparison (BIC) for approach forests')
print('p success containing model:')
print(BIC(mod.appro.pSucc))
print('OP containing model:')
print(BIC(mod.appro.OP))
print('=================================================')
print('Model comparison (BIC) for avoidance forests')
print('p success containing model:')
print(BIC(mod.avoid.pSucc))
print('OP containing model:')
print(BIC(mod.avoid.OP))

# ##### Approach forests
# # Interaction for multivariate p_succ
# mod.appro.pSucc <- glmer(fora_response ~ multiheuristic +
#                            (1 | subject_ID), 
#                          data=df_approach, 
#                          family="binomial"(link = "logit"))
# # isSingular(mod.appro.pSucc)
# summary(mod.appro.pSucc)
# 
# # Interaction for multivariate OP
# mod.appro.OP <- glmer(fora_response ~ OP_cap +
#                         (1 | subject_ID), 
#                       data=df_approach, 
#                       family="binomial"(link = "logit"))
# # isSingular(mod.appro.OP)
# summary(mod.appro.OP)
# 
# ##### Avoidance forests
# # Interaction for multivariate p_succ
# mod.avoid.pSucc <- glmer(fora_response ~ multiheuristic +
#                            (1 | subject_ID), 
#                          data=df_avoidanc, 
#                          family="binomial"(link = "logit"))
# # isSingular(mod.avoid.pSucc)
# summary(mod.avoid.pSucc)
# 
# # Interaction for multivariate OP
# mod.avoid.OP <- glmer(fora_response ~ OP_cap +
#                         (1 | subject_ID), 
#                       data=df_avoidanc, 
#                       family="binomial"(link = "logit"))
# # isSingular(mod.avoid.OP)
# summary(mod.avoid.OP)


print('Model comparison (BIC) for approach forests')
print('p success containing model:')
print(BIC(mod.appro.pSucc))
print('OP containing model:')
print(BIC(mod.appro.OP))
print('=================================================')
print('Model comparison (BIC) for avoidance forests')
print('p success containing model:')
print(BIC(mod.avoid.pSucc))
print('OP containing model:')
print(BIC(mod.avoid.OP))
