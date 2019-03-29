function out                = train_matlabLME(dataTrain, params) 
% Train a MATLAB LME based model 
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


out.dataTrain                   = dataTrain;
out.params                      = params;

P                               = params.P;

n_tasks                         = dataTrain.n_tasks;
targets_cell                    = dataTrain.targets_cell;

assert(length(targets_cell) == n_tasks);

tbl_lme                         = table;
tbl_lme.(dataTrain.name)        = vertcat(targets_cell{:});
tbl_lme.id                      = repVec(dataTrain.ids, dataTrain.nSamples);
tbl_lme.t                       = vertcat(dataTrain.times_cell{:});
for i = 1:size(params.X_vals, 2)
    tbl_lme.(params.X_names{i}) = repVec(params.X_vals(:,i), dataTrain.nSamples);
end

formula                         = sprintf('%s ~ %s', dataTrain.name, params.formula_rhs);
if isfield(params, 'FitMethod')
    fitMethod                   = params.FitMethod;
else
    fitMethod                   = 'ML';
end
post.lme                        = fitlme(tbl_lme, formula, 'FitMethod', fitMethod);

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


