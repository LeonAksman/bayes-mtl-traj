function sim_both_quick_old

addpath '../utils';
addpath '../blr';

%form parameter structure
loopVars.tasks                        	= 100;        	%number of tasks (number of subjects in multi-task trajectory modeling)   
loopVars.observationNoises            	= [1 2 4 8];    %measurement noise standard deviation
loopVars.biomarkerNoises             	= 1;            %biomarker noise standard deviation
loopVars.numRngs                      	= 10;           %number of simulation runs, each a random sampling of trajectories

saveFile_int                            = '../out_blr_sim/int_quick.mat';
saveFile_slope                       	= '../out_blr_sim/slope_quick.mat';

%loop building intercept coupled and slope coupled models over the chosen params
[metrics_intercept, model_names_int,    loopVars]    = loopParams(loopVars, @generatePredictionStructure_intercepts, 'intercept',   saveFile_int);
[metrics_slope,     model_names_slope,  loopVars]    = loopParams(loopVars, @generatePredictionStructure_slopes,     'slope',       saveFile_slope);

assert(isequal(model_names_int, model_names_slope));
model_names                             = model_names_int;

%display figures
createMAEFig(           1, loopVars, metrics_intercept, metrics_slope, model_names);
createCoverageFratioFig(2, loopVars, metrics_intercept, metrics_slope, model_names);
