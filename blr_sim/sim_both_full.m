function sim_both_full

addpath '../utils';
addpath '../blr';
addpath(genpath('../freesurfer_lme/lme'));
addpath(genpath('../gpml-matlab-v4.0-2016-10-19'));

%form parameter structure
loopVars.tasks                        	= 200;        	%number of tasks (number of subjects in multi-task trajectory modeling)   
loopVars.observationNoises            	= [1 2 4 8];    %measurement noise standard deviation
loopVars.biomarkerNoises             	= 1;            %biomarker noise standard deviation
loopVars.numRngs                      	= 30; %50;      %number of simulation runs, each a random sampling of trajectories
loopVars.numTestSamples                 = 1;

saveFile_int                            = '../out_blr_sim/int_full_rerun.mat';
saveFile_slope                       	= '../out_blr_sim/slope_full_rerun.mat';

%loop building intercept coupled and slope coupled models over the chosen params
[metrics_intercept, model_names_int,    loopVars]    = loopParams(loopVars, @generatePredictionStructure_intercepts, 'intercept',   saveFile_int);
[metrics_slope,     model_names_slope,  loopVars]    = loopParams(loopVars, @generatePredictionStructure_slopes,     'slope',       saveFile_slope);

assert(isequal(model_names_int, model_names_slope));
model_names                             = model_names_int;


%********** reorder models
model_names_reorder                     = {'Gaussian both', 'Gaussian int', 'Gaussian slope', 'random',  'linear both', 'linear int', 'linear slope', ...
                                           'plain', 'LME: rI', 'LME: rI,rS', 'OLS'};
assert(isequal(sort(model_names), sort(model_names_reorder)));
assert(isequal(fieldnames(metrics_intercept{1}), fieldnames(metrics_slope{1})));

index_new                               = stringIndex(model_names_reorder, model_names);
fields                                  = fieldnames(metrics_intercept{1});
for i = 1:l(metrics_intercept)
    for j = 1:l(fields)
        metrics_intercept{i}.(fields{j}) = metrics_intercept{i}.(fields{j})(:, index_new);
        metrics_slope{i}.(fields{j})     = metrics_slope{i}.(fields{j})(:, index_new);
    end
end
model_names_intercept               	= model_names_reorder;
model_names_slope                       = model_names_reorder;

%add '*' to oracle models in each scenario
model_names_intercept{2}            	= 'Gaussian int*';
model_names_intercept{6}              	= 'linear int*';
model_names_slope{3}                    = 'Gaussian slope*';
model_names_slope{7}                    = 'linear slope*';

%************* output
%display figures
createMAEFig(           1, loopVars, metrics_intercept, metrics_slope, model_names_intercept, model_names_slope);
createCoverageFratioFig4(2, loopVars, metrics_intercept, metrics_slope, model_names_intercept, model_names_slope);
