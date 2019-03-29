%*******************************************************************************
function plot_models(models, scale, fn_plot_mtl)    

if nargin < 3
    fn_plot_mtl             = @plot_mtl_general;
end

startFig                     = 1;

if isfield(models(1), 'out')
    fields                      = fieldnames(models(1).out);
else
    fields                      = fieldnames(models(1).modelOutput);
end

for i = 1:length(fields)
  
    models_i              	= [];
    for j = 1:length(models)
        models_i(j).name    = models(j).name;
        models_i(j).out   	= models(j).out.(fields{i});
    end
    
    %plot_mtl_general(models_i, startFig + i - 1, fields{i}, scale);
    feval(fn_plot_mtl, models_i, startFig + i - 1, fields{i}, scale);
end
