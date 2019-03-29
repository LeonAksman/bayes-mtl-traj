function plot_mtl_general(data, figureNum, superLabel, scale)%, inputTrainColor, inputTestColor)

addpath '../utils';

SCALE_MAE                       = scale; 

% if nargin < 5
%     inputTrainColor             = 'b';
% end
% if nargin < 6
%     inputTestColor              = 'r';
% end

if ishandle(figureNum)
    close(figureNum);
end

figure(figureNum);

n                               = length(data);

for i = 1:n
    
    name_i                      = data(i).name;
    data_i                      = data(i).out;
    
    train_cell                  = data_i.targets_cell;
    targetsTest_cell            = data_i.targetsTest_cell;
    estimates_cell_mtl          = data_i.estimatesCell_mtl;
    
    subplot(n+1, 1, i);

    additionalString            = '';
    if ~isnan(data_i.p_val)
        additionalString        = sprintf('-log p-val: %.1f', -log10(data_i.p_val));
        %additionalString        = sprintf('p-val: %10.6f', data_i.p_val);
    end
    
    title(sprintf('"%s"  MAE: %.2f, Relative Error: %.2f%%    %s', name_i, data_i.mae_mtl * SCALE_MAE, data_i.relErr_mtl, additionalString));

    hold on;
    if isfield(data_i, 'designMat_cell') && isfield(data_i, 'designMatTest_cell')
        plot_local(train_cell,        train_cell,         data_i.designMat_cell,        '-b.');
        plot_local(targetsTest_cell,  estimates_cell_mtl, data_i.designMatTest_cell,    '-r.');
    else
        plot_local(train_cell,        train_cell,         data_i.times_cell,        '-b.');
        plot_local(targetsTest_cell,  estimates_cell_mtl, data_i.timesTest_cell,    '-r.');    
    end
    hold off;
    
    %**** add independent model at the bottom
    if i == n
        %estimates_cell_indep  	= toTasksCell(data_i.estimatesVec_indep,	data_i.nSamples_test);
        estimates_cell_indep    = data_i.estimatesCell_indep;
        
        subplot(n+1, 1, n+1);

        additionalString            = '';
        if ~isnan(data_i.p_val)
            additionalString        = sprintf('-log p-val: %.1f', -log10(data_i.p_val_indep));
            %additionalString        = sprintf('p-val: %10.6f', data_i.p_val);
        end        
        
        title(sprintf('"Independent"  MAE: %.2f, Relative Error: %.2f%%    %s', data_i.mae_indep * SCALE_MAE, data_i.relErr_indep, additionalString));

        hold on;
        if isfield(data_i, 'designMat_cell') && isfield(data_i, 'designMatTest_cell')
            plot_local(train_cell,        train_cell,           data_i.designMat_cell,       '-b.'); 
            plot_local(targetsTest_cell,  estimates_cell_indep, data_i.designMatTest_cell,   '-r.');  
        else
            plot_local(train_cell,        train_cell,           data_i.times_cell,       '-b.'); 
            plot_local(targetsTest_cell,  estimates_cell_indep, data_i.timesTest_cell,   '-r.'); 
        end
    hold off;
    end
end
      
[~, h1]     = suplabel(superLabel, 't');
set(h1,'FontSize',20)

 %*****************************************************************      
% function plot_local(targetsCell, estimatesCell, designCell, plotParams)% symb)
 function plot_local(targetsCell, estimatesCell, times, plotParams)% symb)

nTasks                      = length(targetsCell);

hold on;
for i = 1:nTasks
    
    targets_i               = targetsCell{i};
    
    if isempty(targets_i)
        continue;
    end
    
    
    estimates_i             = estimatesCell{i};
    
    %design_i                = designCell{i};
    %times_i                 = design_i(:, 2);
    times_i                 = times{i};
    
    plot(times_i, estimates_i, plotParams);  %['-r' symb]);
    plot(times_i, targets_i,   '-b.');%plotParams);  %['-b' symb]);    
end

hold off;

