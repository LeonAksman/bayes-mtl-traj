function test_blr_mtl_GOOD()


NOISE_GLOBAL                    = 3; 
NOISE_LOCAL                     = 10;
coeff_noise_var                 = 0;

%generate samples from noisy polynomial function

n_timepoints                    = 100;
t_final                         = 10;

n_samples                       = 5;
n_samples_test                  = 5;
n_tasks                         = 10;


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

nSamplesPerTaskVec              = zeros(n_tasks, 1);

for i = 1:n_tasks

    %************* TRAINING
    index_i                     = sort(randsample(n_timepoints, n_samples, false), 'ascend');
    n_i                         = length(index_i);
    
    t_i                         = t(index_i);
    nSamplesPerTaskVec(i)       = n_i;
   
   	%create longitudinal design matrix
    Z_i                         = zeros(n_samples, P + 1);
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
    t_i_test                    = t_i(end) + (1:n_samples_test)';
    
    Z_i_test                    = zeros(n_samples_test, P + 1);
    for j = 0:P
        Z_i_test(:, j+1)      	= t_i_test .^ j;
    end    
    targetsTest_cell{i}         = Z_i_test * coeffs_true + normrnd(0, NOISE_LOCAL, n_samples_test, 1);
    designMatTest_cell{i}      	= Z_i_test; 
end

%*** create necessary matrices

targetsTest_cell                = conditional(DEBUG_USE_TRAINING, targets_cell,     targetsTest_cell);
designMatTest_cell             	= conditional(DEBUG_USE_TRAINING, designMat_cell,   designMatTest_cell);

data.P                          = P;
data.n_tasks                   	= n_tasks;
data.nSamplesPerTaskVec         = nSamplesPerTaskVec;

data.targets_cell            	= targets_cell;
data.targetsTest_cell           = targetsTest_cell;
data.designMat_cell             = designMat_cell;
data.designMatTest_cell         = designMatTest_cell;

close all;

figure(1);
trainTestModel(data, @blr_mtl);

figure(2);
trainTestModel(data, @blr_mtl_chol);

figure(3);
trainTestModel(data, @blr_mtl_chol2);


%***************************************************************************
function trainTestModel(data, f_blr)

P                           = data.P;
n_tasks                     = data.n_tasks;
nSamplesPerTaskVec          = data.nSamplesPerTaskVec;

targets_cell                = data.targets_cell;
targetsTest_cell            = data.targetsTest_cell;
designMat_cell              = data.designMat_cell;
designMatTest_cell          = data.designMatTest_cell;

targets_all                 = vertcat(targets_cell{:});
targetsTest_all             = vertcat(targetsTest_cell{:});
designMat_all               = blkdiag(designMat_cell{:});
designMatTest_all           = blkdiag(designMatTest_cell{:});


%********************* MTL
[log_beta, log_alpha, logit_gamma]  = deal(0); 
hyp                                 = [log_beta; log_alpha; logit_gamma];

%Train
maxeval                     = -20;
[hyp,nlmls]                 = minimize_quiet(hyp, f_blr, maxeval, designMat_all, targets_all, n_tasks, P + 1);

%Test
[estimatesVec_mtl, ~, ~] 	= feval(f_blr, hyp, designMat_all, targets_all, n_tasks, P + 1, designMatTest_all); % predictive mean and variance


dispf('nTasks * invLogit(hyp(3)) = %.2f', n_tasks * invLogit(hyp(3)));



%********************* STL
estimatesVec_indep        	= designMatTest_all * pinv(designMat_all) * targets_all;


%********************* plotting

estimates_cell_mtl         	= estimatesToCell(estimatesVec_mtl,     nSamplesPerTaskVec);
estimates_cell_indep       	= estimatesToCell(estimatesVec_indep,   nSamplesPerTaskVec);

subplot(2,1,1);
title('MTL');
plot_mtl(targetsTest_cell, estimates_cell_mtl, designMatTest_cell);

subplot(2,1,2);
title('Independent');
plot_mtl(targetsTest_cell, estimates_cell_indep, designMatTest_cell);


%********************* errors
rmse_mtl                    = sqrt( (estimatesVec_mtl   - targetsTest_all)' * (estimatesVec_mtl   - targetsTest_all) / length(targetsTest_all));
rmse_indep                	= sqrt( (estimatesVec_indep - targetsTest_all)' * (estimatesVec_indep - targetsTest_all) / length(targetsTest_all));

dispf('MTL rmse: %.2f, Indep rmse: %.2f', rmse_mtl, rmse_indep);


%**************************************************
function estimatesCell      = estimatesToCell(estimatesVec, nSamplesPerTaskVec)

nTasks                      = length(nSamplesPerTaskVec);

estimatesCell               = cell(nTasks, 1);
iCurr                       = 1;
for i = 1:nTasks
    n_i                     = nSamplesPerTaskVec(i);
    estimatesCell{i}        = estimatesVec(iCurr:(iCurr + n_i - 1));
    iCurr                   = iCurr + n_i;
end

%**************************************************
function plot_mtl(targetsCell, estimatesCell, designCell)

nTasks                      = length(targetsCell);

hold on;
for i = 1:nTasks
    
    targets_i               = targetsCell{i};
    estimates_i             = estimatesCell{i};
    
    design_i                = designCell{i};
    times_i                 = design_i(:, 2);
    
    plot(times_i, targets_i);
    plot(times_i, estimates_i, '-r');
end

hold off;
