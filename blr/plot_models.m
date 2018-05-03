%*******************************************************************************
function plot_models(models, scale)    

startFig                     = 1;

fields                      = fieldnames(models(1).out);

for i = 1:length(fields)
  
    models_i              	= [];
    for j = 1:length(models)
        models_i(j).name    = models(j).name;
        models_i(j).out   	= models(j).out.(fields{i});
    end
    
    plot_mtl_general(models_i, startFig + i - 1, fields{i}, scale);
end
