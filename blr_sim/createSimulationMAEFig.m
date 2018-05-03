function createSimulationMAEFig

addpath '../utils';
addpath '../aboxplot';
addpath '../tightfigadv';

dataDir                 = '~/Documents/MATLAB/out_blr_sim'; 
loadFilename_intercepts = fullfile(dataDir, 'sim_interceptCoupling_200_varyObsNoise.mat');
loadFilename_slopes     = fullfile(dataDir, 'sim_slopeCoupling_200_varyObsNoise.mat');

load(loadFilename_intercepts);
metrics_intercept    	= metrics_all; 

load(loadFilename_slopes); 
metrics_slope        	= metrics_all;

model_names{3}          = 'Gaussian both';
model_names{5}          = 'Gaussian int';
model_names{7}          = 'Gaussian slope';


createMAEFig(loopVars, metrics_intercept, metrics_slope, model_names);
