function out                = train_fsLME(dataTrain, params) 
% Train a Freesurfer LME based model 
%
% Inputs: 
%   dataTrain.n_tasks:          number of tasks (e.g. subjects)
%   dataTrain.targets_cell:     cell of size n_tasks containing targets vector for each task
%   dataTrain.designMat_cell:   cell of size n_tasks containing design matrix for each task
%
%   params.normDesignMatrix:    boolean: true if normalizing each column of design matrix
%   params.normTargets:         boolean: true if normalizing targets
%
%   params.P:                   model order of fitted polynomial functions
%                               (1 for linear trajectories, 2 for quadratic, etc.)     


out.dataTrain               = dataTrain;
out.params                  = params;

P                           = params.P;

n_tasks                     = dataTrain.n_tasks;
targets_cell                = dataTrain.targets_cell;
designMat_cell              = dataTrain.designMat_cell;

assert(length(targets_cell) == n_tasks & length(designMat_cell) == n_tasks);

%******** prepare fixed effects matrix
fixedEffects_mat            = [];
for i = 1:size(params.fixedEffects, 2)
    fixedEffects_mat     	= [fixedEffects_mat repVec(params.fixedEffects(:,i), dataTrain.nSamples)];
end

%normalize design matrix
if params.normDesignMatrix
    [X_train, stats_design]	= normalizeDesignMat_fsLME(designMat_cell, fixedEffects_mat);
else
    X_train              	= vertcat(designMat_cell{:});
    X_train                 = [X_train fixedEffects_mat];
end

%normalize targets
targets_all                 = vertcat(targets_cell{:});
if params.normTargets
    [targets_all, stats_targets] = standardize(targets_all);
end

y_train                     = targets_all; 

assert(all(~isnan(X_train(:))));
post.fs_stats            	= lme_fit_FS(X_train, params.randomEffectsVec, y_train, dataTrain.nSamples);

n_randomEffects            	= length(params.randomEffectsVec);
post.b                     	= repmat(post.fs_stats.Bhat', n_tasks, 1);
post.b(:, 1:n_randomEffects) = post.b(:, 1:n_randomEffects) + post.fs_stats.bihat';

%***** NEW
%assumes fixed effects are in order: 1, t, t^2, ..., t^P, then other covariates
post.invA                	= repmat(diag(post.fs_stats.CovBhat(1:(P+1), 1:(P+1))), n_tasks, 1);
covBi_new                 	= zeros(n_tasks, n_randomEffects); 
for i = 1:n_randomEffects
    covBi_new(:, i)         = post.fs_stats.Covbihat(i:n_randomEffects:end, i);
end
for i = 1:min(n_randomEffects, P+1)
    post.invA(i:(P+1):end)  = post.invA(i:(P+1):end) + covBi_new(:, i);
end

post.Dhat                   = post.fs_stats.Dhat;  
post.phisqhat               = post.fs_stats.phisqhat;  

%rescale back coefficients based on target scaling
if params.normTargets
    post.b                  = rescale_coeffs_b(post.b,        stats_targets);
    
    post.invA            	= post.invA              * (stats_targets.sDev^2);  
    
    %**** used for calculating variance of predictions
    post.Dhat            	= post.fs_stats.Dhat  	 * (stats_targets.sDev^2);  
    post.phisqhat           = post.fs_stats.phisqhat * (stats_targets.sDev^2);  
end

%rescale the design matrix after modelling
if params.normDesignMatrix
    
    %assert(isequal(params.randomEffectsVec, 1:(P+1)), ...
    assert(all(ismember(params.randomEffectsVec, 1:(P+1))), ...
           'randomEffectsVec must be a subset of 1:(P+1) in LME model. Turn off normDesignMatrix flag to avoid this error.');
    
    %*** rescale coeffs
    % in normalized variables: y = a * x1_norm + b * x2_norm + c  
    % in original   variables: y = a/sigma1 * x1 + b/sigma2 * x2 + [c - a*mu1/sigma1 - b*mu2/sigma2]
    % ... can be generalized to n variables trivially
    %
    n_col                    	= size(post.b, 2);

    for i = 2:n_col
        int_update_i          	= (post.b(:, i) * stats_design.mean(i))/stats_design.std(i);
        
        %update intercepts
     	%post.m(1:(P+1):end)   	= post.m(1:(P+1):end)           - int_update_i; 
        post.b(:, 1)          	= post.b(:, 1) - int_update_i;
        
        %update current coeffs
        %post.m(i:(P+1):end)    	= post.m(i:(P+1):end)         	/ designMat_std(i);
        post.b(:, i)         	= post.b(:, i) / stats_design.std(i);
        
    end
  
    covScaler                   = stats_design.std;
    assert(covScaler(1) == 0);
    covScaler(1)                = 1;
    covScaler                   = 1 ./ (covScaler .* covScaler');

    covScaler_RE                = covScaler(params.randomEffectsVec,  params.randomEffectsVec);
    post.Dhat                   = post.fs_stats.Dhat * covScaler_RE;
    
    covScaler_Pplus1            = covScaler(1:(P+1), 1:(P+1));
    covScaler_rep               = repmat(diag(covScaler_Pplus1), n_tasks, 1);
    post.invA                   = diag(diag(post.invA) .* covScaler_rep);
end

post.m                          = post.b(:, 1:(P+1))';
post.m                          = post.m(:);

out.post                        = post;


%*********************************************************
function [trainOut, stats]     	= standardize(trainIn)

[stats.m, stats.sDev]         	= deal(mean(trainIn), std(trainIn));
trainOut                      	= (trainIn - stats.m)/stats.sDev;

%*********************************************************
function outVec                         = rescale_coeffs_b(inVec, stats)

outVec                                  = zeros(size(inVec));
outVec(:, 1)                            = inVec(:, 1)	* stats.sDev + stats.m;

nCols                                   = size(inVec, 2);
for i = 2:nCols
    outVec(:, i)                        = inVec(:, i) * stats.sDev;
end

%*********************************************************
function [designMat_vert, stats_design]  = normalizeDesignMat_fsLME(designMat_cell, fixedEffects_mat)

designMat_vert                          = vertcat(designMat_cell{:});

%**** freesurfer fixed effects
%for i = 1:size(params.fixedEffects, 2)
%    designMat_vert                      = [designMat_vert fixedEffects_mat(:,i)]; %repVec(params.fixedEffects(:,i), nSamples_train)];
%end
designMat_vert                          = [designMat_vert fixedEffects_mat];


[stats_design.mean, stats_design.std]   = deal(mean(designMat_vert), std(designMat_vert));
assert(unique(designMat_vert(:, 1)) == 1);

for i = 2:size(designMat_vert, 2)  
    designMat_vert(:, i)             	= (designMat_vert(:, i) - stats_design.mean(i)) ./ stats_design.std(i);
end


