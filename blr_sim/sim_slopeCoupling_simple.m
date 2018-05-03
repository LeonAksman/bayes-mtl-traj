function sim_slopeCoupling_simple

addpath '../utils';
addpath '../blr';


loopVars.tasks                        	= 50;   %number of tasks (subjects for trajectory modeling)   
loopVars.observationNoises            	= 1;    %measurement noise standard deviation
loopVars.biomarkerNoises             	= 1;    %biomarker noise standard deviation
loopVars.numRngs                      	= 10;   %number of simulation runs, each a random sampling of trajectories
 
saveFilename                            = '..\out_blr_sim\sim_slopeCoupling_simple.mat';
[metrics_all, model_names, loopVars]    = loopParams(saveFilename, loopVars, @generatePredictionStructure_slopes, 'slope');


createCoverageFratioFig(loopVars, metrics_intercept, metrics_slope, model_names);
