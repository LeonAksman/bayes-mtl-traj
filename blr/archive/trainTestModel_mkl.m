function out                = trainTestModel_mkl(data, targetCol, f_blr)

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


%********************* MTL
%numHyp                      = 3 + length(extraKernels);
%numHyp                      = 2 + n_tasks + length(extraKernels);
%hyp                      	= zeros(numHyp, 1);

%Train
maxeval                     = -20;
%tic;
[hyp, ~]                    = minimize_quiet(hyp, f_blr, maxeval, designMat_all, targets_all, n_tasks, P + 1, extraKernels);
%toc;

%***** to get negative log marginal likelihood of training data
[nlZ, ~]                    = feval(f_blr, hyp, designMat_all, targets_all, n_tasks, P + 1, extraKernels);

% if options.verbose
%   
%     alpha1                  = exp(hyp(2));
%     alpha2                  = exp(hyp(3));
%     dispf('***** alpha1 (indep): %.4f, alpha2 (coupled): %.4f, log ratio: alpha2/alpha1: %.1f', alpha1, alpha2, log(alpha2/alpha1));
% 
% end


%Test
[estimatesVec_mtl, ~, post]	= feval(f_blr, hyp, designMat_all, targets_all, n_tasks, P + 1, extraKernels, designMatTest_all); % predictive mean and variance

%********************* output
out.logML                   = -nlZ;
%out.coeffs                	= post.m;

%********************* STL
estimatesVec_indep        	= designMatTest_all * pinv(designMat_all) * targets_all;

out.err_mtl                 = estimatesVec_mtl   - targetsTest_all;
out.err_indep               = estimatesVec_indep - targetsTest_all;

out.targets_all             = targets_all;
out.targetsTest_all         = targetsTest_all;

out.designMat_cell        	= data.designMat_cell;
out.designMatTest_cell   	= data.designMatTest_cell;

out.estimatesVec_mtl        = estimatesVec_mtl;
out.estimatesVec_indep      = estimatesVec_indep;

% %********************* errors
err_mtl                     = estimatesVec_mtl   - targetsTest_all;
err_indep                   = estimatesVec_indep - targetsTest_all;

out.nSamples_train        	= data.nSamples_train;
out.nSamples_test          	= data.nSamples_test;

out.targets_all             = targets_all;
out.targetsTest_all         = targetsTest_all;

out.estimatesVec_mtl        = estimatesVec_mtl;
out.estimatesVec_indep      = estimatesVec_indep;

out.mae_mtl               	= sum(abs(err_mtl))   / length(targetsTest_all);
out.mae_indep             	= sum(abs(err_indep)) / length(targetsTest_all);

out.relErr_mtl              = sum(abs(err_mtl))   / sum(abs(targetsTest_all)) * 100;
out.relErr_indep            = sum(abs(err_indep)) / sum(abs(targetsTest_all)) * 100;

p_val_sign_rank             = signrank(abs(err_mtl), abs(err_indep));
out.p_val                  	= conditional(out.mae_mtl < out.mae_indep, p_val_sign_rank, NaN);

% 
% %********************** significance testing
% %paired t-test on errors
% %[~, p_test]             	= ttest(abs(err_mtl), abs(err_indep)); 
% p_test                      = signrank(abs(err_mtl), abs(err_indep));
% 
% %dispf('Mean |err mtl|: %.5f, mean |err indep|: %.5f, mean diff: %.5f, signed rank p-val: %.4f', ... % paired t-test p-val: %.4f', ...
% %            mean(abs(err_mtl)), mean(abs(err_indep)), mean(abs(err_mtl)) - mean(abs(err_indep)), p_test);
% 
%  
