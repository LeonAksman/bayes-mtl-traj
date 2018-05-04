function simple_example
% Generate some intercept-varying trajectories and build several different
% types of models, visualizing the results
%
addpath '../utils';
addpath '../blr';

if nargin < 1
    modelOutfile            = '';
end

if ~isempty(modelOutfile) && fileExist(modelOutfile)
    dispf('Found %s, loading models from file.', modelOutfile);
    load(modelOutfile);
    plot_models(models);
    return;
else
    dispf('Failed to find %s, recomputing models.', modelOutfile);
end

n_tasks                         = 200;

[predictStruct, traj_coeffs]    = generatePredictionStructure(n_tasks, 8);

intercepts                      = traj_coeffs(:, 1);
slopes                          = traj_coeffs(:, 2);
similarityKernel                = squareform(pdist(intercepts, 'squaredeuclidean'));

%***** random
rng(1);
randvec                         = rand(n_tasks, 1);
random_gauss                    = squareform(pdist(randvec, 'squaredeuclidean'));
random_gauss                    = random_gauss / norm(random_gauss);

similarityKernel_linear         = computeSimilarityKernel(intercepts); %using a noiseless biomarker here
kernelSubjectIds                = unique(predictStruct.subj_id);

%*************** default params
MIN_SAMPLES_TRAINING            = 2;
commonParams.P                  = 1;
commonParams.mode               = 'predict_last';
commonParams.minTrainingSamples = MIN_SAMPLES_TRAINING;
commonParams.extraKernels       = [];
commonParams.f_blr              = @blr_mtl_mkl_inv;
commonParams.f_optimizer      	= @minimize;
commonParams.maxeval            = -100;             %
commonParams.standardize        = true;
commonParams.kernelSubjectIds   = kernelSubjectIds;

%**************** specify models
models                          = [];

clear model;

%random gaussian based kernel
model.name               	= 'random';
model.params              	= commonParams;
model.params.extraKernels(1).mat        = random_gauss;
model.params.extraKernels(1).type    	= 'gaussian';   
model.params.extraKernels(1).bound    	= 'positive'; 
models                      = [models; model];

%linear kernel based coupling
model.name               	= 'linear kernel coupled';
model.params              	= commonParams;
model.params.extraKernels(1).mat        = similarityKernel_linear;
model.params.extraKernels(1).type    	= 'linear';   
model.params.extraKernels(1).bound    	= 'positive'; 
models                      = [models; model];

%gaussian kernel based coupling
model.name               	= 'gaussian kernel coupled';
model.params              	= commonParams;
model.params.extraKernels(1).mat        = similarityKernel;
model.params.extraKernels(1).type    	= 'gaussian';   
model.params.extraKernels(1).bound    	= 'positive'; 
models                      = [models; model];

%multi-kernel: linear plus gaussian kernels
model.name               	= 'multi';
model.params              	= commonParams;
model.params.extraKernels(1).mat        = similarityKernel_linear;
model.params.extraKernels(1).type    	= 'linear';   
model.params.extraKernels(1).bound    	= 'positive'; 
model.params.extraKernels(2).mat        = similarityKernel;
model.params.extraKernels(2).type    	= 'gaussian';   
model.params.extraKernels(2).bound    	= 'positive'; 
models                      = [models; model];

%simple coupling: uncoupled and fully coupled
clear model;
model.name               	= 'plain';
model.params              	= commonParams;
models                      = [models; model];

SCALE_MAE                  	= 1;

%***** compute and plot
for i = 1:length(models)
    dispf('**** running model: %s', models(i).name);
    models(i).out          	= predict_blr_mtl_flex(predictStruct, models(i));
    dispf('**********************');
    
end

for i = 1:length(models)
    int_i                   = models(i).out.sim.post.m(1:2:end);
    slope_i                 = models(i).out.sim.post.m(2:2:end);
    
    all_stds                = sqrt(models(i).out.sim.post.invA); %sqrt(diag(models(i).out.sim.post.invA));

    NUM_STDS                = 2;
    
    %*** compute intercepts coverage: fraction of times intercept is within 2 stds of true value 
    int_i_std               = all_stds(1:2:end);
    is_good_int_i       	= intercepts >= (int_i - NUM_STDS*int_i_std) & intercepts <= (int_i + NUM_STDS*int_i_std);
  
    %*** compute slope coverage
    slope_i_std          	= all_stds(2:2:end);
    is_good_slope_i       	= slopes >= (slope_i - NUM_STDS*slope_i_std) & slopes <= (slope_i + NUM_STDS*slope_i_std);

    dispf('%30s: INTERCEPTS coverage: %.2f, SLOPES coverage: %.2f', models(i).name, ...
                                                           sum(is_good_int_i)/length(is_good_int_i), ...
                                                           sum(is_good_slope_i)/length(is_good_slope_i));
end

plot_models(models, SCALE_MAE);
compare_models(models);


%***************************************************************
function [predictStruct, traj_coeffs] = generatePredictionStructure(n_tasks, rngSeed)

rng(rngSeed)

%100 timepoints between 0 and 10
n_timepoints                    = 100;
t_start                         = 0;
t_end                           = 10;
t                               = linspace(t_start, t_end, n_timepoints)';

%3 samples per subject
n_samples_per_task              = 3; 

%5 timepoint steps between measurement for each subject
STEP                            = 5;  
MAX_TRAINING_INDEX              = n_timepoints - (STEP * n_samples_per_task);

n_samples_vec                	= n_samples_per_task * ones(n_tasks, 1);

%linear models
modelP                          = 1;
Z                               = [ones(n_timepoints, 1) t]; 

%measurement noise
NOISE_GLOBAL                    = 0.5;   

%varying intercepts, fixed slopes across subjects
offsets                        = -10:2:10;
traj_coeffs                    = zeros(n_tasks, 2);
traj_coeffs(:, 1)              = randsample(offsets,   n_tasks, true);
traj_coeffs(:, 2)              = -1;                                    

subj_id                         = [];
[age, simVals]                  = deal([]);

%run through each task (subject), generate measurements to fill out
%prediction structure
for i = 1:n_tasks
    
    n_samples_i                 = n_samples_vec(i);
    
    subj_id_i                   = repmat(i, n_samples_i, 1); %repmat({sprintf('S_%d', i)}, n_samples_i, 1);
    
       
    index_i1                    = randsample(1:MAX_TRAINING_INDEX, 1, false);
    index_i                     = [index_i1:STEP:(index_i1 + STEP*n_samples_i - 1)];    
    
    
    age_i                       = t(index_i);
    coeffs_i                    = traj_coeffs(i, :)';
    simVals_i                   = Z(index_i, :) * coeffs_i + normrnd(0, NOISE_GLOBAL, n_samples_i, 1);
    
    subj_id                     = [subj_id;     subj_id_i];
    age                         = [age;         age_i];
    simVals                     = [simVals;     simVals_i];
    
end


predictStruct.subj_id         	= subj_id;
predictStruct.age               = age;
predictStruct.age_raw          	= age;
predictStruct.sim               = simVals;

