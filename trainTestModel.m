function out                = trainTestModel(data, targetCol, params) 

f_blr                       = params.f_blr;
f_optimizer                 = params.f_optimizer;
if isfield(params, 'maxeval')
    maxeval                 = params.maxeval;
else
    maxeval                 = -2000;
end

P                           = data.P;
n_tasks                     = data.n_tasks;
nSamples_train            	= data.nSamples_train;
nSamples_test               = data.nSamples_test;

targets_cell                = data.targets_cell;
targetsTest_cell            = data.targetsTest_cell;

designMat_cell              = data.designMat_cell;
designMatTest_cell          = data.designMatTest_cell;

extraKernels                = data.extraKernels;

targets_all                 = vertcat(targets_cell{:});
targetsTest_all             = vertcat(targetsTest_cell{:});

targets_all                 = targets_all(:,        targetCol);
targetsTest_all             = targetsTest_all(:,    targetCol);

designMat_all               = blkdiag(designMat_cell{:});
designMatTest_all           = blkdiag(designMatTest_cell{:});



%****************** rescale by standardizing

if params.standardize
    [targets_all, ...
     targetsTest_all,stats]	= standardize(targets_all, targetsTest_all);
end


%************************************* OLS
indep.m                     = pinv(designMat_all) * targets_all;
estimatesVec_indep        	= designMatTest_all * indep.m;

%************************************* BLR

%**************** Train
isEmpty                     = data.isEmpty;
hyp                         = data.hyp;
%profile on;
[hyp, fX, i]             	= feval(f_optimizer, hyp, f_blr, maxeval, designMat_all, targets_all, n_tasks, P + 1, extraKernels);
%profile viewer;

%***** to get negative log marginal likelihood of training data
[nlZ, ~]                    = feval(f_blr, hyp, designMat_all, targets_all, n_tasks, P + 1, extraKernels);

%**************** Test
[estimatesVec_mtl, estimatesVec_mtl_var, post]	= feval(f_blr, hyp, designMat_all, targets_all, n_tasks, P + 1, extraKernels, designMatTest_all); % predictive mean and variance

%**************** rescale back
if params.standardize
    estimatesVec_indep     	= rescale(estimatesVec_indep,   stats);
    estimatesVec_mtl       	= rescale(estimatesVec_mtl,     stats);
    estimatesVec_mtl_var  	= estimatesVec_mtl_var * (stats.sDev^2); % rescale(estimatesVec_mtl_var,     stats);
    
    post.m                  = rescale_coeffs(post.m,        stats,  P);
    post.invA            	= post.invA * (stats.sDev^2);                   %rescale(post.invA,          	stats);
    indep.m                 = rescale_coeffs(indep.m,   	stats,  P);
end

%**************** output
out.indep                   = indep;
out.post                    = post;
out.hyp                     = hyp;
out.logML                   = -nlZ;

out.designMat_cell        	= data.designMat_cell;
out.designMatTest_cell   	= data.designMatTest_cell;
out.targets_cell           	= data.targets_cell;

%******* testing cells
n_empty                     = length(find(isEmpty == 1));
targetsTest_cell(isEmpty == 1)  = cell(n_empty, 1);
out.targetsTest_cell            = targetsTest_cell;

%*******
%     targetsTest_cell            = data_i.targetsTest_cell;
%     estimates_cell_mtl          = data_i.estimatesCell_mtl;
%*******

[out.estimatesCell_mtl, ...
 out.estimatesCell_indep] 	= deal(cell(n_tasks, 1));

%********** prev
%estimatesVec_indep_nz    	= estimatesVec_indep(~isEmpty);
%estimatesVec_mtl_nz       	= estimatesVec_mtl(~isEmpty);

%********** new
nSamples_test_mod           = nSamples_test;
nSamples_test_mod(nSamples_test == 0) = 1;
estimates_cell_mtl         	= toTasksCell(estimatesVec_mtl,         nSamples_test);      %nSamples_test_mod);  %
estimatesVec_mtl_nz       	= estimates_cell_mtl(~isEmpty); 
estimatesVec_mtl_nz         = vertcat(estimatesVec_mtl_nz{:});
%***var
estimatesCell_mtl_var    	= toTasksCell(estimatesVec_mtl_var,     nSamples_test);  %nSamples_test_mod);  %
estimatesVec_mtl_var_nz    	= estimatesCell_mtl_var(~isEmpty); 
estimatesVec_mtl_var_nz    	= vertcat(estimatesVec_mtl_var_nz{:});

estimates_cell_indep        = toTasksCell(estimatesVec_indep,       nSamples_test);     %nSamples_test_mod);%
estimatesVec_indep_nz     	= estimates_cell_indep(~isEmpty); 
estimatesVec_indep_nz     	= vertcat(estimatesVec_indep_nz{:});

iCurr                       = 1;
for i = 1:n_tasks
    
    if isEmpty(i)
        continue;
    end
    
    n_i                     = nSamples_test(i);
    index_i                 = iCurr:(iCurr + n_i - 1);
    iCurr                   = iCurr + n_i;
    
 	out.estimatesCell_mtl{i}        = estimatesVec_mtl_nz(index_i);
    out.estimatesCell_mtl_var{i}    = estimatesVec_mtl_var_nz(index_i);
    out.estimatesCell_indep{i}      = estimatesVec_indep_nz(index_i);
end
%********************


out.isEmpty                 = isEmpty;

out.nSamples_train        	= data.nSamples_train;
out.nSamples_test          	= data.nSamples_test;

% out.targets_all             = targets_all;
% out.targetsTest_all         = targetsTest_all;
% out.estimatesVec_mtl        = estimatesVec_mtl;
% out.estimatesVec_indep      = estimatesVec_indep;

% %********************* statistics, evalutated at non-zero test targets only

%***** prev
%estimatesVec_indep          = estimatesVec_indep(~isEmpty);
%estimatesVec_mtl            = estimatesVec_mtl(~isEmpty);
%targetsTest_all             = targetsTest_all(~isEmpty);
%***** new
estimatesVec_indep          = estimatesVec_indep_nz; 
estimatesVec_mtl            = estimatesVec_mtl_nz;
targetsTest_nz              = targetsTest_cell(~isEmpty);
targetsTest_all             = vertcat(targetsTest_nz{:});


out.err_mtl                	= estimatesVec_mtl   - targetsTest_all;
out.err_indep              	= estimatesVec_indep - targetsTest_all;

out.mae_mtl               	= sum(abs(out.err_mtl))   / length(targetsTest_all);
out.mae_indep             	= sum(abs(out.err_indep)) / length(targetsTest_all);

out.relErr_mtl              = sum(abs(out.err_mtl))   / sum(abs(targetsTest_all)) * 100;
out.relErr_indep            = sum(abs(out.err_indep)) / sum(abs(targetsTest_all)) * 100;

%p_val_sign_rank             = signrank(abs(out.err_mtl), abs(out.err_indep));
%out.p_val                  	= conditional(out.mae_mtl < out.mae_indep, p_val_sign_rank, NaN);
out.p_val                   = signrank(estimatesVec_mtl,    targetsTest_all);
out.p_val_indep             = signrank(estimatesVec_indep,  targetsTest_all);

%*********************************************************
function [trainOut, testOut, stats]  	= standardize(trainIn, testIn)

[stats.m, stats.sDev]                 	= deal(mean(trainIn), std(trainIn));

trainOut                                = (trainIn - stats.m)/stats.sDev;
testOut                                 = (testIn  - stats.m)/stats.sDev;

%*********************************************************
function outVec                         = rescale(inVec, stats)

outVec                                  = inVec * stats.sDev + stats.m;

%*********************************************************
function outVec                         = rescale_coeffs(inVec, stats, P)

outVec                                  = zeros(size(inVec));
outVec(1:(P+1):end)                     = inVec(1:(P+1):end)	* stats.sDev + stats.m;
for i = 2:(P+1)
    outVec(i:(P+1):end)                 = inVec(i:(P+1):end)    * stats.sDev;
end


