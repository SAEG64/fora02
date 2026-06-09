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
df_allSubs <- read_csv("DATA_clean/DATA_group_level/test_data.group_level_datall.csv")
# # Add WWS model and bin_e
df_allSubs['ternary'] <- df_allSubs['ternary state']
df_allSubs['p_succ'] <- df_allSubs['** $\\mathit{p}$ success']
df_allSubs['OP_values'] <- df_allSubs['optimal policy values']
df_allSubs['condition'] <- df_allSubs['condition_rORp']
# df_allSubs['OP_cap'] <- df_allSubs["OP_cap"]

# Interaction for multivariate p_succ
mod.2var.pSucc <- glmer(fora_response ~ factor(ternary) +
                    factor(condition) *
                    scale(p_succ) +
                    (1 | subject_ID),
                  data=df_allSubs,
                  family="binomial"(link = "logit"),
                  control = glmerControl(optimizer="bobyqa"))
# isSingular(mod.2var.pSucc)
summary(mod.2var.pSucc)

# Interaction for multivariate OP
mod.2var.OP <- glmer(fora_response ~ factor(ternary) +
                          factor(condition) *
                          scale(OP_values) +
                          (1 | subject_ID),
                        data=df_allSubs,
                        family="binomial"(link = "logit"),
                        control = glmerControl(optimizer="bobyqa"))
# isSingular(mod.2var.OP)
summary(mod.2var.OP)

# Full model
mod <- glmer(fora_response ~ (factor(ternary) +
                                scale(OP_values) + 
                                scale(p_succ)) *
                          factor(condition) +
                          (1 | subject_ID),
                        data=df_allSubs,
                        family="binomial"(link = "logit"),
                        control = glmerControl(optimizer="bobyqa"))
# isSingular(mod)
summary(mod)

# OP BIC:     10832.0
# p succ BIC: 10379.9

## Predictions for plot
#ggpredict(m3, terms="p_succ_correct [all]")
#ggpredict(m3, terms = "r_threat")
#gg <- ggpredict(
#  mdl, terms = c(terms="p_succ_correct [all]", "r_threat", "condition_rORp"))
## Create plot
#ggplot(gg, aes(x = x, y = predicted, colour = group)) +
#  geom_line() +
#  facet_wrap(~facet) +
#  labs(
#    title = "Predicted foraging likelihoods for two conditions:\nforests with low and high amount of threats",
#       x = ~italic(p)~"success corrected", y = "Foraging likelihood")+
#  guides(fill = guide_legend(title="Threat risk (bins)"), 
#         color = guide_legend(title="Threat risk (bins)"))+
#  theme_bw() +
#  theme(axis.text.x = element_text(size=16),
#        axis.text.y = element_text(size=16),
#        axis.title.x = element_text(size = 20),
#        axis.title.y = element_text(size = 20),
#        title = element_text(size=18),
#        legend.text = element_text(size=18))

