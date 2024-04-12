# Analyses of the paper "Multi-Heuristic Policies in a Sequention Decision-Making Task"
The raw data was preprocessed with the fora0_prep_data_singe.py and fora0_prep_data_group.py script. Download entire folder including all the data folders to run the following analyses:
- Compute MDP values
- Model comparisons
- Model fits
- Mixed effects models
- Model correlations
- Response time analysis
- Parameter recovery and confusion matrix

## Calculate the PEP values by your self
- PEP plots are generated from precalculated values.
- If you want to calculate the PEP values by yourself, the fora_logit_BIC_and_BF.py script needs to be run first and then run the fora_PEP_values.m script.
- You can select a data subset (experimental conditions) in the fora_logit_BIC_and_BF.py in order to run fora_PEP_values.m for this subset data.
- In order to plot the newly calculated PEP values, you then must change the PEP output file (csv) in the results folder to match the corresponding name in the fora_logit_PEP_plot.py script.

## Extract the MDP model values
- You need to index into the correct cell in the generated MDP_action_value_difference.csv file, since the forest order shown to participants was randomized.
- Therefore, you need to get the forest Nr (column 'fora_Nr' in the subjects' data sheets) and compute the correct row index by using  the function mentioned in end of the fora_MDP.py script.
- The row number in the MDP_action_value_differenc.csv file represents the time-points in an environment.
- The column number in the MDP_action_value_differenc.csv file represents the energy state of the participant (column 'in_LP' in the subjects' data sheets).
