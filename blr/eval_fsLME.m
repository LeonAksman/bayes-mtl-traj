function out                = eval_fsLME(predictStruct, targets_cell) 

[targets,   ...
 preds_lme, ...
 predVars_lme]            	= deal([]);

n_tasks                     = length(targets_cell);
for i = 1:n_tasks
    
    if isempty(targets_cell{i})
        continue;
    end
    
    targets                 = [targets;         targets_cell{i}];
    preds_lme               = [preds_lme;       predictStruct.predictions_lme{i}];
    predVars_lme            = [predVars_lme;    predictStruct.predictionVars_lme{i}];
end

out.targets                 = targets;

out.preds                   = preds_lme;
out.predVars                = predVars_lme;

out.err                     = conditional(isempty(targets), 0, preds_lme - targets);
out.mae                     = conditional(isempty(targets), 0, sum(abs(out.err)) / length(targets));