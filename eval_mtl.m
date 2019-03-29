function out                = eval_mtl(predictStruct, targets_cell) 

[targets,       ...
 preds_ols,     ...
 predVars_ols,  ...
 preds_mtl,     ...
 predVars_mtl]              = deal([]);

n_tasks                     = length(targets_cell);
for i = 1:n_tasks
    
    if isempty(targets_cell{i})
        continue;
    end
    
    targets                 = [targets;         targets_cell{i}];
    preds_mtl               = [preds_mtl;       predictStruct.predictions_mtl{i}];
    preds_ols               = [preds_ols;       predictStruct.predictions_ols{i}];

    predVars_mtl            = [predVars_mtl;    predictStruct.predictionVars_mtl{i}];
    predVars_ols            = [predVars_ols;    predictStruct.predictionVars_ols{i}];
end

out.targets                 = targets;

out.preds                   = preds_mtl;
out.preds_ols               = preds_ols;

out.predVars                = predVars_mtl;
out.predVars_ols            = predVars_ols;

%var_residuals               = ((targets - preds_ols) .* (targets - preds_ols)')/(length(targets) - 2);
%out.predVars_ols            = var_residuals * 

out.err_ols                 = conditional(isempty(targets), 0, preds_ols - targets);
out.err                     = conditional(isempty(targets), 0, preds_mtl - targets);

out.mae_ols                 = conditional(isempty(targets), 0, sum(abs(out.err_ols)) / length(targets));
out.mae                     = conditional(isempty(targets), 0, sum(abs(out.err))   	 / length(targets));
