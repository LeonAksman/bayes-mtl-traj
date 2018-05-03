function test_blr_mtl_v2()

addpath '../utils';
addpath(genpath('../PRoNTo_v.1.1_r740/machines/gpml/gpml-v3.1'));

NOISE_GLOBAL                    = 5; %3; 
NOISE_LOCAL                     = 10;
coeff_noise_var                 = 0;

%generate samples from noisy polynomial function

n_timepoints                    = 100;
t_final                         = 10;

n_tasks                         = 100;

min_samples                     = 1;
max_samples                     = 3;

n_samples                       = randsample(min_samples:max_samples, n_tasks, true);
n_samples_test                  = randsample(min_samples:max_samples, n_tasks, true);


DEBUG_USE_TRAINING              = false;
if DEBUG_USE_TRAINING
    disp('DEBUG: Evaluating predictions at TRAINING points.');
else
    disp('Evaluating predictions at TESTING points.');
end
%*******************************************

t                               = linspace(0, t_final, n_timepoints)';



Z                               = [ones(n_timepoints, 1) t t .^2];
coeffs_true                     = [10; 5; 1];

P                               = 2;
Z                               = Z(:, 1:(P+1));
coeffs_true                     = coeffs_true(1:(P + 1));


targetsAll                    	= Z * coeffs_true + normrnd(0, NOISE_GLOBAL, n_timepoints, 1);

%training
[targets_cell, designMat_cell]          = deal(cell(n_tasks, 1));
[targetsTest_cell, designMatTest_cell]  = deal(cell(n_tasks, 1));

[nSamples_train, nSamples_test]     	= deal(zeros(n_tasks, 1));

for i = 1:n_tasks
    
   	nSamples_train(i)           = conditional(length(n_samples) == 1,       n_samples,      n_samples(i));
    nSamples_test(i)           	= conditional(length(n_samples_test) == 1,  n_samples_test, n_samples_test(i));

    %************* TRAINING
    index_i                     = sort(randsample(n_timepoints, nSamples_train(i), false), 'ascend');
    n_i                         = length(index_i);
    
    t_i                         = t(index_i);
   
   	%create longitudinal design matrix
    Z_i                         = zeros(nSamples_train(i), P + 1);
    for j = 0:P
        Z_i(:, j+1)             = t_i .^ j;
    end
    targets_i               	= targetsAll(index_i) + normrnd(0, NOISE_LOCAL, n_i, 1);
    
    %for BLR
    designMat_cell{i}         	= Z_i; 
    targets_cell{i}            	= targets_i;    
    
    %************* TESTING
    
   	%subsample from remaining time-points
    %index_remain_i              = setdiff(1:n_timepoints, index_i);
    %index_i_test                = sort(randsample(index_remain_i, n_samples_test, false), 'ascend');       
    %t_i_test                    = t(index_i_test); 
    t_i_test                    = t_i(end) + (1:nSamples_test(i))';
    
    Z_i_test                    = zeros(nSamples_test(i), P + 1);
    for j = 0:P
        Z_i_test(:, j+1)      	= t_i_test .^ j;
    end    
    targetsTest_cell{i}         = Z_i_test * coeffs_true + normrnd(0, NOISE_LOCAL, nSamples_test(i), 1);
    designMatTest_cell{i}      	= Z_i_test; 
end

%*** create necessary matrices

targetsTest_cell                = conditional(DEBUG_USE_TRAINING, targets_cell,     targetsTest_cell);
designMatTest_cell             	= conditional(DEBUG_USE_TRAINING, designMat_cell,   designMatTest_cell);

data.P                          = P;
data.n_tasks                   	= n_tasks;
data.nSamples_train             = nSamples_train;
data.nSamples_test              = nSamples_test;

data.targets_cell            	= targets_cell;
data.targetsTest_cell           = targetsTest_cell;
data.designMat_cell             = designMat_cell;
data.designMatTest_cell         = designMatTest_cell;

tt_options.verbose              = true;
tt_options.plot                 = true;

close all;

% figure(1);
% trainTestModel(data, 1, @blr_mtl, tt_options);
% 
% figure(2);
% trainTestModel(data, 1, @blr_mtl_chol, tt_options);
% 

figure(1);
tt_options.figureNum            = 1;
trainTestModel(data, 1, @blr_mtl_chol2, tt_options);

figure(2);
tt_options.figureNum            = 2;
trainTestModel(data, 1, @blr_mtl_mkl, tt_options);
