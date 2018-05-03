function [predictStruct, traj_coeffs, true_diffs] = generatePredictionStructure_intercepts(params)

[n_tasks, rngSeed, observationNoise] = deal(params.n_tasks, params.rngSeed, params.observationNoise);

rng(rngSeed)

n_timepoints                    = 100;
t_start                         = 0;
t_end                           = 10;
t                               = linspace(t_start, t_end, n_timepoints)';

n_samples_per_task              = 3;

STEP                            = 5;  
MAX_TRAINING_INDEX              = n_timepoints - (STEP * n_samples_per_task);

n_samples_vec                	= n_samples_per_task * ones(n_tasks, 1);


modelP                          = 1;
Z                               = [ones(n_timepoints, 1) t];

true_diffs                     = randsample(-10:2:-2,   n_tasks, true); %****-20:4:-4 , -10:5:10

traj_coeffs                     = zeros(n_tasks, 2);
traj_coeffs(:, 1)               = true_diffs;               
traj_coeffs(:, 2)               = -1;           % + normrnd(0, paramNoise, 1, n_tasks);


subj_id                         = []; %{};
[age, simVals]                  = deal([]);

for i = 1:n_tasks
    
    n_samples_i                 = n_samples_vec(i);
    
    subj_id_i                   = repmat(i, n_samples_i, 1);%repmat({sprintf('S_%d', i)}, n_samples_i, 1);
    
       
    index_i1                    = randsample(1:MAX_TRAINING_INDEX, 1, false);
    index_i                     = [index_i1:STEP:(index_i1 + STEP*n_samples_i - 1)];    
    
    %NUM_EXTRA_STEPS             = 1;
    %index_i(end)                = min(n_timepoints, index_i(end) + STEP*NUM_EXTRA_STEPS);
    
    age_i                       = t(index_i);
    coeffs_i                    = traj_coeffs(i, :)';
    simVals_i                   = Z(index_i, :) * coeffs_i + normrnd(0, observationNoise, n_samples_i, 1);
    
    subj_id                     = [subj_id;     subj_id_i];
    age                         = [age;         age_i];
    simVals                     = [simVals;     simVals_i];
    
end


predictStruct.subj_id         	= subj_id;
predictStruct.unique_subj_id    = (1:n_tasks)';
predictStruct.age               = age;
predictStruct.age_raw          	= age;
predictStruct.sim               = simVals;