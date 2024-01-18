# Analyses of the paper "Multi-Heuristic Policies in a Sequention Decision-Making Task"
- Compute MDP values
- Model comparisons
- Model fits
- Mixed effects models
- Model correlations
- Response time analysis
- Parameter recovery and confusion matrix

## Calculate the PEP values by your self
- PEP plots are generated from precalculated values.
- If you want to calculate the PEP values by your self, the fora_logit_BIC_and_BF.py script needs to be run first.
- You can select a data subset (experimental conditions) in the fora_logit_BIC_and_BF.py in order to run fora_PEP_values.m for this subset data.
- To run the fora_PEP_values.m for a data you computed, you will have to change the BIC file name in the input of the fora_PEP_values.m script to the file name you created when running the fora_logit_BIC_and_BF.py script.

## Extract the MDP model values
- You need to index into the correct cell in the generated MDP_action_value_difference.csv file, since the forest order shown to participants was randomized.
- Therefore, you need to get the forest Nr (column 'fora_Nr' in the subjects' data sheets) and compute the correct row index by using  the function mentioned in end of the fora_MDP.py script.
- The column number in the MDP_action_value_difference.csv file represents the energy state of the participant (column 'in_LP' in the subjects' data sheets).
