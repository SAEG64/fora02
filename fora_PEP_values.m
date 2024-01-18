% Execute this after running the python script "BIC_and_log-GroupBF"
% Choose a condition when executing BIC_and_log-GroupBF to get subset PEP

% Set working directory
tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));

% Load and prepare data
data = dlmread('RESULTS/fora_BICs.csv',',',1,0);
data = -0.5*data;
% Load spm
spm; % (make sure to add spm to path)
% Compute PEP
[alpha1,exp_r1,xp1,pxp1,bor1] = spm_BMS(data); 
% Export results
csvwrite('RESULTS/fora_PEPs.csv',pxp1);

