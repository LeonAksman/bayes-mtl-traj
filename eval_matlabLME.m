function out                = eval_matlabLME(predictStruct, targets_cell) 

[targets,   ...
 preds_lme]                 = deal([]);

n_tasks                     = length(targets_cell);
for i = 1:n_tasks
    
    if isempty(targets_cell{i})
        continue;
    end
    
    targets                 = [targets;     targets_cell{i}];
    preds_lme               = [preds_lme;   predictStruct.predictions_lme{i}];
end

out.err_lme                 = conditional(isempty(targets), 0, preds_lme - targets);
out.mae_lme                 = conditional(isempty(targets), 0, sum(abs(out.err_lme)) / length(targets));