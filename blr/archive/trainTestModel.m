function out                = trainTestModel(data, targetCol, f_blr, options)

P                           = data.P;
n_tasks                     = data.n_tasks;
nSamples_train            	= data.nSamples_train;
nSamples_test               = data.nSamples_test;

targets_cell                = data.targets_cell;
targetsTest_cell            = data.targetsTest_cell;
designMat_cell              = data.designMat_cell;
designMatTest_cell          = data.designMatTest_cell;

targets_all                 = vertcat(targets_cell{:});
targetsTest_all             = vertcat(targetsTest_cell{:});

targets_all                 = targets_all(:,        targetCol);
targetsTest_all             = targetsTest_all(:,    targetCol);

designMat_all               = blkdiag(designMat_cell{:});
designMatTest_all           = blkdiag(designMatTest_cell{:});


%********************* MTL
[log_beta, log_alpha, logit_gamma]  = deal(0); 
hyp                                 = [log_beta; log_alpha; logit_gamma];

%Train
maxeval                     = -20;
%tic;
[hyp,nlmls]                 = minimize_quiet(hyp, f_blr, maxeval, designMat_all, targets_all, n_tasks, P + 1);
%toc;

%***** to get negative log marginal likelihood of training data
[out.nlZ, ~]              	= feval(f_blr, hyp, designMat_all, targets_all, n_tasks, P + 1);


if options.verbose
  
%     invAlpha                = 1/exp(hyp(2));
%     gamma                   = invLogit(hyp(3));
%     
%     indep_term              = invAlpha * n_tasks * gamma;
%     coupled_term            = invAlpha * (1 - gamma);
%     dispf('indep term: %.2f, coupled term: %.2f', indep_term, coupled_term);

    alpha1                  = exp(hyp(2));
    alpha2                  = exp(hyp(3));
    dispf('***** alpha1 (indep): %.4f, alpha2 (coupled): %.4f, log ratio: alpha2/alpha1: %.1f', alpha1, alpha2, log(alpha2/alpha1));

end


%Test
[estimatesVec_mtl, ~, post]	= feval(f_blr, hyp, designMat_all, targets_all, n_tasks, P + 1, designMatTest_all); % predictive mean and variance

%********************* output
out.coeffs                	= post.m;

if ~options.plot
    return;
end


%********************* STL
estimatesVec_indep        	= designMatTest_all * pinv(designMat_all) * targets_all;


%********************* errors
err_mtl                     = estimatesVec_mtl   - targetsTest_all;
err_indep                   = estimatesVec_indep - targetsTest_all;

%rmse_mtl                    = sqrt( err_mtl'   * err_mtl    / length(targetsTest_all));
%rmse_indep                	= sqrt( err_indep' * err_indep  / length(targetsTest_all));

mae_mtl                     = sum(abs(err_mtl))   / length(targetsTest_all);
mae_indep                   = sum(abs(err_indep)) / length(targetsTest_all);

mae_mtl_percent             = sum(abs(err_mtl))   / sum(abs(targetsTest_all)) * 100;
mae_indep_percent       	= sum(abs(err_indep)) / sum(abs(targetsTest_all)) * 100;

%********************* plotting
train_cell                  = toTasksCell(targets_all,          nSamples_train);

estimates_cell_mtl         	= toTasksCell(estimatesVec_mtl,     nSamples_test);
estimates_cell_indep       	= toTasksCell(estimatesVec_indep,   nSamples_test);

targetsTest_cell            = toTasksCell(targetsTest_all,  	nSamples_test);

if ishandle(options.figureNum)
    close(options.figureNum);
end


figure(options.figureNum);

subplot(2,1,1);

% title(sprintf('MTL:         RMSE %.5f, MAE: %.5f, relMAE: %.1f%%', rmse_mtl, mae_mtl, mae_mtl_percent));
title(sprintf('MTL:         MAE: %.1f, relMAE: %.1f%%', mae_mtl * 1e3, mae_mtl_percent));

hold on;
plot_mtl(train_cell,        train_cell,         designMat_cell,        '.');
plot_mtl(targetsTest_cell,  estimates_cell_mtl, designMatTest_cell,    '.');
hold off;

subplot(2,1,2);

% title(sprintf('Independent: RMSE %.5f, MAE: %.5f, relMAE: %.1f%%', rmse_indep, mae_indep, mae_indep_percent));
title(sprintf('Independent: MAE: %.1f, relMAE: %.1f%%', mae_indep *  1e3, mae_indep_percent));


hold on;
plot_mtl(train_cell,       train_cell,           designMat_cell,        '.');
plot_mtl(targetsTest_cell, estimates_cell_indep, designMatTest_cell,    '.');
hold off;

%paired t-test on errors
%[~, p_test]             	= ttest(abs(err_mtl), abs(err_indep)); 
p_test                      = signrank(abs(err_mtl), abs(err_indep));

dispf('Mean |err mtl|: %.5f, mean |err indep|: %.5f, mean diff: %.5f, signed rank p-val: %.4f', ... % paired t-test p-val: %.4f', ...
            mean(abs(err_mtl)), mean(abs(err_indep)), mean(abs(err_mtl)) - mean(abs(err_indep)), p_test);

%**************************************************
function plot_mtl(targetsCell, estimatesCell, designCell, sym)

nTasks                      = length(targetsCell);

hold on;
for i = 1:nTasks
    
    targets_i               = targetsCell{i};
    estimates_i             = estimatesCell{i};
    
    design_i                = designCell{i};
    times_i                 = design_i(:, 2);
    
    plot(times_i, estimates_i, ['-r' sym]);
    plot(times_i, targets_i,   ['-b' sym]);    
end

hold off;
