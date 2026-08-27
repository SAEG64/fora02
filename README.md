# Context-Dependent Feature Reweighting in Sequential Approach-Avoidance Decisions

This repository contains code and behavioral data associated with **Study 2** of the doctoral dissertation:

**Values, Features, and Beliefs in Adaptive Sequential Decision-Making**  
Sergej Golowin  
Heidelberg University, 2026

## Overview

This study investigates how environmental context changes the information used during sequential decision-making.

Participants completed a sequential foraging task in which they chose between **foraging** and **waiting** while maintaining an energy reserve. Foraging involved probabilistic food rewards and probabilistic predator threats.

The task contained **approach** and **avoidance** environments that differed in the relationship between reward and threat.

The main analyses tested whether contextual adaptation was better characterized by:

- changes in alignment with an MDP-derived optimal policy,
- or selective reweighting of decision-relevant task features within a shared policy architecture.

The optimal policy (MDP_policy.csv) and corresponding state-action values (MDP_action_values.csv) were precomputed using backwards induction. Features refer to the expression levels of environmental compponents (e.g. weather type, probability of successful foraging, etc.) The results support context-dependent feature reweighting rather than a switch between qualitatively distinct decision policies.

## Participants

The study included **29 participants**.

Each participant completed 72 sequential foraging environments with up to eight decision trials per environment.

The final dataset contained **10,270 analyzed decisions**.

## Main Analyses

The reported analyses include:

- computation of the normative MDP-derived action-value difference,
- hierarchical Bayesian behavioral modeling,
- comparison of feature-based and optimal-policy predictors,
- context-dependent moderation of decision features,
- model recovery,
- posterior predictive checks,
- control analyses of task difficulty and response times.

The main decision variables included:

- reward probability,
- threat probability,
- participant energy state,
- wait-when-safe state,
- MDP-derived Delta Q.

## Main Result

Behavior reflected the combined influence of multiple decision features.

Avoidance contexts reduced the influence of reward-related information while increasing alignment with the integrated MDP-derived value signal.

Overall, the results indicate that approach and avoidance contexts changed the relative weighting of decision-relevant information rather than producing a global switch to a different decision strategy.

## Requirements

The analyses are written primarily in Python.

Main packages include:

```text
numpy
pandas
scipy
matplotlib
seaborn
pymc
arviz
scikit-learn
statsmodels
```

## Running the Repository

Clone the repository:

```bash
git clone https://github.com/SAEG64/fora02.git
cd fora02
```

Install the required Python dependencies and run the Study 2 analysis scripts corresponding to the analyses reported in the dissertation.

## Citation

Golowin, S. (2026).  
**Values, Features, and Beliefs in Adaptive Sequential Decision-Making.**  
Doctoral dissertation, Heidelberg University.
