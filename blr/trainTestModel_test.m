function out                = trainTestModel_test(data, targetCol, f_blr, f_optimizer)

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

isEmpty                     = data.isEmpty;

hyp                         = data.hyp;

%**************** Train
maxeval                     = -2000; 
%[hyp, fX, i]             	= feval(f_optimizer, hyp, f_blr, maxeval, designMat_all, targets_all, n_tasks, P + 1, extraKernels);
hyp = [0; 0];
[hyp, fX, i]                = feval(f_optimizer, hyp, f_blr, maxeval, designMat_all, targets_all);  %[varargout] = blr(hyp, X, t, xs)

%***** to get negative log marginal likelihood of training data
%[nlZ, ~]                    = feval(f_blr, hyp, designMat_all, targets_all, n_tasks, P + 1, extraKernels);
[nlZ, ~]                    = feval(f_blr, hyp, designMat_all, targets_all);

dispf('alpha: %f', hyp(2));

%**************** Test
%[estimatesVec_mtl, ~, post]	= feval(f_blr, hyp, designMat_all, targets_all, n_tasks, P + 1, extraKernels, designMatTest_all); % predictive mean and variance
[estimatesVec_mtl, ~, post]	= feval(f_blr, hyp, designMat_all, targets_all, designMatTest_all); % predictive mean and variance

%********************* output
out.post                    = post;
out.hyp                     = hyp;
out.logML                   = -nlZ;
%out.coeffs                	= post.m;

%********************* STL
out.indep.m                 = pinv(designMat_all) * targets_all;
estimatesVec_indep        	= designMatTest_all * out.indep.m;

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
estimates_cell_mtl         	= toTasksCell(estimatesVec_mtl,    nSamples_test_mod);  %nSamples_test);
estimatesVec_mtl_nz       	= estimates_cell_mtl(~isEmpty); 
estimatesVec_mtl_nz         = vertcat(estimatesVec_mtl_nz{:});

estimates_cell_indep        = toTasksCell(estimatesVec_indep, 	nSamples_test_mod);%nSamples_test);
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
    
 	out.estimatesCell_mtl{i}    = estimatesVec_mtl_nz(index_i);
    out.estimatesCell_indep{i}  = estimatesVec_indep_nz(index_i);
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


err_mtl                     = estimatesVec_mtl   - targetsTest_all;
err_indep                   = estimatesVec_indep - targetsTest_all;

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
