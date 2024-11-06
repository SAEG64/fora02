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
df_allSubs <- read_csv("DATA_clean/DATA_group_level/datall_with_condition_order.csv")
# Add WWS model and bin_e
df_allSubs['bin_e_state'] <- df_allSubs['** binary energy state']
df_allSubs['wait_when_safe'] <- df_allSubs['** wait when safe']
df_allSubs['OP_values'] <- df_allSubs['optimal policy values']
df_allSubs['MHP_model'] <- df_allSubs['multi-heuristic policy']
df_allSubs['OP_cap'] <- df_allSubs["OP_cap"]
# Dummy code condition order
#data_dummy <- data.frame(df_allSubs[ , ! colnames(df_allSubs) %in% "x1"], model.matrix( ~ cond_order - 1, df_allSubs))
# Run mixed effects model
mdl <- glmer(
  fora_response ~ 
    OP_cap * cond_order +
    (1|subject_ID), 
  data=df_allSubs, family="binomial"(link = "logit"), 
  control = glmerControl(optimizer="bobyqa"))
ss <- getME(mdl,c("theta","fixef"))
#ss <- getME(mdl,"ALL")
m3 <- update(
  mdl,start=ss,control=glmerControl(
    optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
summary(m3)
# Post hoc test
library(multcomp)
df_allSubs$sxt <- interaction(df_allSubs$OP_cap,df_allSubs$cond_order)
glm.posthoc <- glm(fora_response~-1 + sxt, data=df_allSubs,family="binomial", weights=rep(10, nrow(df_allSubs)))
summary(glht(glm.posthoc, mcp(sxt="Tukey")))





dfbetas <- as.data.frame(dfbetas(m3))


library(lmerTest)
aov <- anova(m3)
ft = m3.fit()
ranova(m3)

library(effects)
my.eff <- Effect(c("OP_cap", "cond_order"), m3)
plot(my.eff)
summary(my.eff)

# Post-hoc analysis
library(emmeans)
emt1 <- emtrends(m3, "condition_rORp", var = "OP_cap")
emt1          ### estimated slopes of ZcNOF for each factor level
pairs(emt1)   ### comparison of slopes
