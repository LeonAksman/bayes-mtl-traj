%*******************************************************************************
function plot_models_new(models, scale)    

if nargin < 3
    fn_plot_mtl             = @plot_mtl_general;
end

fieldNames               	= fieldnames(models(1).modelOutput);
modelNames               	= {models.name};

for i = 1:length(fieldNames)
  
    if ishandle(i)
        close(i);
    end
    figure(i);

    n                               = length(modelNames);
    
    for j = 1:n
      
        dataTrain                   = models(j).modelOutput.(fieldNames{i}).dataTrain;
        
        if isfield(models(j).modelOutput.(fieldNames{i}), 'logML')
            logML_ij               	= models(j).modelOutput.(fieldNames{i}).logML;
        else
            logML_ij                = 0;
        end
        
        predStruct_ij               = models(j).predictStruct.(fieldNames{i});
        dataTest                    = predStruct_ij.dataTest;
        
        evalStruct_ij               = models(j).evalOutput.(fieldNames{i});
        
        train_cell                  = dataTrain.targets_cell;
        targetsTest_cell            = dataTest.targets_cell;
        
        if      isfield(predStruct_ij, 'predictions_mtl')
            estimates_cell_i        = predStruct_ij.predictions_mtl;
            mae_i                   = evalStruct_ij.mae_mtl;
        elseif isfield(predStruct_ij, 'predictions_lme')
            estimates_cell_i       	= predStruct_ij.predictions_lme;
            mae_i                   = evalStruct_ij.mae_lme;
        else
            error('Cannot find MTL or LME predictions for model %d', j);
        end
        
        subplot(n+1, 1, j);

        title(sprintf('"%s"  MAE: %.3f, logML %.1f', models(j).name, mae_i * scale, logML_ij));

        hold on;
        %if isfield(data_i, 'designMat_cell') && isfield(data_i, 'designMatTest_cell')
        %    plot_local(train_cell,        train_cell,         dataTrain.designMat,          '-b.');
        %    plot_local(targetsTest_cell,  estimates_cell_mtl, dataTest.designMat,           '-r.');
        %else
        plot_local(train_cell,        train_cell,         dataTrain.times_cell,       	'-b.');
        plot_local(targetsTest_cell,  estimates_cell_i,  dataTest.times_cell,    	'-r.');    
        %end
        hold off;

        %**** add independent model at the bottom
        if j == n
            
            estimates_cell_indep   	= predStruct_ij.predictions_ols;
            
            subplot(n+1, 1, n+1);

            title(sprintf('"Independent"  MAE: %.3f', evalStruct_ij.mae_ols * scale));

            hold on;
            %if isfield(data_i, 'designMat_cell') && isfield(data_i, 'designMatTest_cell')
            %    plot_local(train_cell,        train_cell,         dataTrain.designMat,          '-b.');
            %    plot_local(targetsTest_cell,  estimates_cell_indep, dataTest.designMat,      	'-r.');
            %else
            plot_local(train_cell,        train_cell,           dataTrain.times_cell, 	'-b.');
            plot_local(targetsTest_cell,  estimates_cell_indep, dataTest.times_cell,  	'-r.');    
            %end
            hold off;
        end        
    end
    
    [~, h1]     = suplabel(fieldNames{i}, 't');
    set(h1,'FontSize',20)   
end


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
