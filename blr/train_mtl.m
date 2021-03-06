function out                = train_mtl(dataTrain, params) 
% Train a multi-task learning (MTL) based model 
%
% Inputs: 
%   dataTrain.n_tasks:          number of tasks (e.g. subjects)
%   dataTrain.targets_cell:     cell of size n_tasks containing targets vector for each task
%   dataTrain.designMat_cell:   cell of size n_tasks containing design matrix for each task
%
%   params.f_blr:               function handle for MTL inference function called by optimizer
%   params.f_optimizer:         function handle for optimizer
%   params.maxeval:             maximum number of function evalations during optimization
%   params.normDesignMatrix:    boolean: true if normalizing each column of design matrix
%   params.normTargets:         boolean: true if normalizing targets
%
%   params.P:                   model order of fitted polynomial functions
%                               (1 for linear trajectories, 2 for quadratic, etc.)     
%   params.extraKernels:        
%
% Outputs:


out.dataTrain               = dataTrain;
out.params                  = params;

f_blr                       = params.f_blr;
f_optimizer                 = params.f_optimizer;
if isfield(params, 'maxeval')
    maxeval                 = params.maxeval;
else
    maxeval                 = -2000;
end

P                           = params.P;
extraKernels                = params.extraKernels;

n_tasks                     = dataTrain.n_tasks;
targets_cell                = dataTrain.targets_cell;
designMat_cell              = dataTrain.designMat_cell;

assert(length(targets_cell) == n_tasks & length(designMat_cell) == n_tasks);

% ********************* Scale
%normalize design matrix
if params.normDesignMatrix
    [designMat_cell, ...
     stats_design]          = normalizeDesignMat(designMat_cell, P);
end
designMat_all               = blkdiag(designMat_cell{:});

%normalize targets
targets_all                 = vertcat(targets_cell{:});
if params.normTargets
    [targets_all, stats_targets] = standardize(targets_all);
end

% ******************* OLS Train
indep.m                     = pinv(designMat_all) * targets_all;
XtX                         = designMat_all' * designMat_all;
if det(XtX) < 1e-10
    indep.invA             = inv(XtX + 1e-6*eye(size(XtX)));
else
    indep.invA              = inv(XtX);
end

% ******************** MTL Train
validFns                    = {'blr_mtl_mkl_inv', 'blr_mtl_mkl_inv_reindex', 'blr_mtl_mkl_inv_split'};
assert(ismember(func2str(params.f_blr), validFns));
nHyp                        = 2 + (2 + numHyp_kernel(params.extraKernels)) * (P+1); 
hyp                         = zeros(nHyp, 1);

paramsTraining.nTasks       = n_tasks;
paramsTraining.nBlocks      = P + 1;
paramsTraining.noiseMin     = 10e-6;
passed                      = false;
for i = 1:10
    try
        [hyp, fX, i]       	= feval(f_optimizer, hyp, f_blr, maxeval, designMat_all, targets_all, paramsTraining, extraKernels);
        passed              = true;
    catch
        dispf('Model train failed with noiseMin parameter value: %f', paramsTraining.noiseMin);
    end
    if passed
        break;
    end
    paramsTraining.noiseMin = paramsTraining.noiseMin * 10;
end
assert(passed, 'Failed to train model after 10 increases of noise');

% get negative log marginal likelihood of training data
[nlZ, ~, post]          	= feval(f_blr, hyp, designMat_all, targets_all, paramsTraining, extraKernels);

%save the unadjusted posterior variables, which may be
%rescaled below by both target and design matrix related factors
indep_unadjusted            = indep;
post_unadjusted             = post;

% ********************* Rescale back
 if params.normTargets   
  	
    %** OLS 
    indep.m                 = rescale_mean(indep.m,   	stats_targets,  P);
    indep.invA              = indep.invA * (stats_targets.sDev^2);

    %** MTL
    post.m                  = rescale_mean(post.m,        stats_targets,  P);
    post.invA            	= post.invA * (stats_targets.sDev^2);     
    post.beta               = post.beta * 1/(stats_targets.sDev^2); %inverse of noise
    
    
    %scale the alphas (covariance hyperparams)
    %post.alphas              = post.alphas * (stats_targets.sDev^2);    
    %post.hyp(1)             = log(post.beta);
end

%compute mean +/- 2 standard deviations - useful for simuation
indep.twoStd_pos            = indep.m + 2 * sqrt(diag(indep.invA));
indep.twoStd_neg         	= indep.m - 2 * sqrt(diag(indep.invA));

post.twoStd_pos             = post.m  + 2 * sqrt(diag(post.invA));
post.twoStd_neg             = post.m  - 2 * sqrt(diag(post.invA));

% scale       = stats_design.std;
% scale(1)    = 1;
% %scaleCell   = repmat({diag(1./(scale.^2))}, n_tasks, 1);
% scaleCell   = repmat({diag(1./(scale))}, n_tasks, 1);
% scaleMat    = blkdiag(scaleCell{:});
% post.invA   = scaleMat' * post.invA * scaleMat;

%invSigma_scaled = post.invSigma * (stats_targets.sDev^2);
%post.invA   = inv(post.beta * XtX + invSigma_scaled);

%rescale back design matrix and coefficients
if params.normDesignMatrix
    covAdjustment                       = ones(n_tasks * (P+1), 1);
    
    for i = 2:(P+1)
    	%****************** OLS
        int_update_i_indep            	= (indep.m(i:(P+1):end) * stats_design.mean(i))/stats_design.std(i);
        
        %update intercepts
        indep.m(1:(P+1):end)          	= indep.m(1:(P+1):end) - int_update_i_indep;
        indep.twoStd_pos(1:(P+1):end)   = indep.twoStd_pos(1:(P+1):end) - int_update_i_indep;
        indep.twoStd_neg(1:(P+1):end)   = indep.twoStd_neg(1:(P+1):end) - int_update_i_indep;

        %update current coeffs
        indep.m(i:(P+1):end)             = indep.m(i:(P+1):end)             / stats_design.std(i);
        indep.twoStd_pos(i:(P+1):end)    = indep.twoStd_pos(i:(P+1):end)	/ stats_design.std(i);
        indep.twoStd_neg(i:(P+1):end)  	 = indep.twoStd_neg(i:(P+1):end)	/ stats_design.std(i);
        
        %****************** MTL
        int_update_i                    = (post.m(i:(P+1):end) * stats_design.mean(i))/stats_design.std(i);
        
        %update intercepts
        post.m(1:(P+1):end)             = post.m(1:(P+1):end)           - int_update_i;  
        post.twoStd_pos(1:(P+1):end)    = post.twoStd_pos(1:(P+1):end)  - int_update_i;
        post.twoStd_neg(1:(P+1):end)    = post.twoStd_neg(1:(P+1):end)  - int_update_i;

        %update current coeffs
        post.m(i:(P+1):end)             = post.m(i:(P+1):end)         	/ stats_design.std(i);
        post.twoStd_pos(i:(P+1):end)  	= post.twoStd_pos(i:(P+1):end)	/ stats_design.std(i);
        post.twoStd_neg(i:(P+1):end)  	= post.twoStd_neg(i:(P+1):end)	/ stats_design.std(i);

        covAdjustment(i:(P+1):end)       = 1/(stats_design.std(i));  %1/(stats_design.std(i).^2);
    end    
    covAdjustment_mat                   = covAdjustment * covAdjustment';
    indep.invA                          = indep.invA .* covAdjustment_mat;
    post.invA                           = post.invA  .* covAdjustment_mat;
    
end

out.stats_targets           = stats_targets;
out.stats_design            = stats_design;
out.indep_unadjusted      	= indep_unadjusted;
out.post_unadjusted         = post_unadjusted;
out.indep                   = indep;
out.post                    = post;
out.hyp                     = hyp;
out.logML                   = -nlZ;


%*********************************************************
function [trainOut, stats]              = standardize(trainIn)

[stats.m, stats.sDev]                 	= deal(mean(trainIn), std(trainIn));

trainOut                                = (trainIn - stats.m)/stats.sDev;

%*********************************************************
function outVec                         = rescale_mean(inVec, stats, P)

outVec                                  = zeros(size(inVec));
outVec(1:(P+1):end)                     = inVec(1:(P+1):end)	* stats.sDev + stats.m;
for i = 2:(P+1)
    outVec(i:(P+1):end)                 = inVec(i:(P+1):end)    * stats.sDev;
end

