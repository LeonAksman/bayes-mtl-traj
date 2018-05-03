function out                = trainTestModel_v2(data, targetCol, f_blr, options)

P                           = data.P;
n_tasks                     = data.n_tasks;
%nSamples_train            	= data.nSamples_train;
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

if options.verbose

    %Sigma               = kron((1/alpha) * ((1 - gamma) * ones(nTasks) + nTasks * gamma * eye(nTasks)), eye(nDimsPerTask));

    
    invAlpha                = 1/exp(hyp(2));
    gamma                   = invLogit(hyp(3));
    
    indep_term              = invAlpha * n_tasks * gamma;
    coupled_term            = invAlpha * (1 - gamma);
    dispf('indep term: %.2f, coupled term: %.2f', indep_term, coupled_term);
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
rmse_mtl                    = sqrt( (estimatesVec_mtl   - targetsTest_all)' * (estimatesVec_mtl   - targetsTest_all) / length(targetsTest_all));
rmse_indep                	= sqrt( (estimatesVec_indep - targetsTest_all)' * (estimatesVec_indep - targetsTest_all) / length(targetsTest_all));



%********************* plotting
estimates_cell_mtl         	= toTasksCell(estimatesVec_mtl,     nSamples_test);
estimates_cell_indep       	= toTasksCell(estimatesVec_indep,   nSamples_test);

targetsTest_cell            = toTasksCell(targetsTest_all,  	nSamples_test);

if ishandle(options.figureNum)
    close(options.figureNum);
end


figure(options.figureNum);

subplot(2,1,1);
title(sprintf('MTL: RMSE %.1f', rmse_mtl));
plot_mtl(targetsTest_cell, estimates_cell_mtl, designMatTest_cell);

subplot(2,1,2);
title(sprintf('Independent: RMSE %.1f', rmse_indep));
plot_mtl(targetsTest_cell, estimates_cell_indep, designMatTest_cell);



%dispf('MTL rmse: %.2f, Indep rm%se: %.2f', rmse_mtl, rmse_indep);



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
