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
df_allSubs <- read_csv("DATA_clean/DATA_group_level/test_data.group_level.csv")
# Add WWS model and bin_e
df_allSubs['wait_when_safe'] <- NaN
df_allSubs['bin_e_state'] <- df_allSubs['binary energy state']
for (i in 1:nrow(df_allSubs)) {
  if (df_allSubs[i,'in_LP'] > 8-df_allSubs[i,'day.thisN']) {
    df_allSubs[i,'wait_when_safe'] <- 0
  } else {
    df_allSubs[i,'wait_when_safe'] <- 1
  }
}
# Run mixed effects model
mdl <- glmer(
  fora_response ~ 
    p_succ_correct*condition_rORp + 
    r_threat*condition_rORp + 
    wait_when_safe*condition_rORp + 
    bin_e_state*condition_rORp + 
    (1|subject_ID), 
  data=df_allSubs, family="binomial"(link = "logit"), 
  control = glmerControl(optimizer="bobyqa"))
ss <- getME(mdl,c("theta","fixef"))
#ss <- getME(mdl,"ALL")
m3 <- update(
  mdl,start=ss,control=glmerControl(
    optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
summary(m3)
# Predictions for plot
ggpredict(m3, terms="p_succ_correct [all]")
ggpredict(m3, terms = "r_threat")
gg <- ggpredict(
  mdl, terms = c(terms="p_succ_correct [all]", "r_threat", "condition_rORp"))
# Create plot
ggplot(gg, aes(x = x, y = predicted, colour = group)) +
  geom_line() +
  facet_wrap(~facet) +
  labs(
    title = "Predicted foraging likelihoods for two conditions:\nforests with low and high amount of threats",
       x = ~italic(p)~"success corrected", y = "Foraging likelihood")+
  guides(fill = guide_legend(title="Threat risk (bins)"), 
         color = guide_legend(title="Threat risk (bins)"))+
  theme_bw() +
  theme(axis.text.x = element_text(size=16),
        axis.text.y = element_text(size=16),
        axis.title.x = element_text(size = 20),
        axis.title.y = element_text(size = 20),
        title = element_text(size=18),
        legend.text = element_text(size=18))
