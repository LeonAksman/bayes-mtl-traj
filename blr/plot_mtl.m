function plot_mtl(data, figureNum)

train_cell                  = toTasksCell(data.targets_all,        	data.nSamples_train);

estimates_cell_mtl         	= toTasksCell(data.estimatesVec_mtl,  	data.nSamples_test);
estimates_cell_indep       	= toTasksCell(data.estimatesVec_indep,	data.nSamples_test);

targetsTest_cell            = toTasksCell(data.targetsTest_all,     data.nSamples_test);

if ishandle(figureNum)
    close(figureNum);
end

figure(figureNum);

subplot(2,1,1);

% title(sprintf('MTL:         RMSE %.5f, MAE: %.5f, relMAE: %.1f%%', rmse_mtl, mae_mtl, mae_mtl_percent));
title(sprintf('MTL:  MAE: %.2f, Relative Error: %.2f%%', data.mae_mtl * 1e3, data.relErr_mtl));

hold on;
plot_local(train_cell,        train_cell,         data.designMat_cell,        '.');
plot_local(targetsTest_cell,  estimates_cell_mtl, data.designMatTest_cell,    '.');
hold off;

subplot(2,1,2);

% title(sprintf('Independent: RMSE %.5f, MAE: %.5f, relMAE: %.1f%%', rmse_indep, mae_indep, mae_indep_percent));
title(sprintf('Independent: MAE: %.2f, Relative Error: %.2f%%', data.mae_indep *  1e3, data.relErr_indep));


hold on;
plot_local(train_cell,       train_cell,           data.designMat_cell,        '.');
plot_local(targetsTest_cell, estimates_cell_indep, data.designMatTest_cell,    '.');
hold off;
       
 %*****************************************************************      
 function plot_local(targetsCell, estimatesCell, designCell, symb)

nTasks                      = length(targetsCell);

hold on;
for i = 1:nTasks
    
    targets_i               = targetsCell{i};
    estimates_i             = estimatesCell{i};
    
    design_i                = designCell{i};
    times_i                 = design_i(:, 2);
    
    plot(times_i, estimates_i, ['-r' symb]);
    plot(times_i, targets_i,   ['-b' symb]);    
end

hold off;

