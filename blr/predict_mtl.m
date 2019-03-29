function out                = predict_mtl(dataTest, modelOutput) 

out.dataTest                = dataTest;

n_tasks                     = dataTest.n_tasks;
designMatTest_cell          = dataTest.designMat_cell;

assert(length(designMatTest_cell) == n_tasks);
assert(length(modelOutput.post.beta) == 1);

P                           = modelOutput.params.P;

targets_train               = vertcat(modelOutput.dataTrain.targets_cell{:});
X_train                     = blkdiag(modelOutput.dataTrain.designMat_cell{:});
preds_ols_train             = X_train * modelOutput.indep.m;
var_residuals               = sum((targets_train - preds_ols_train).^2)/(length(targets_train) - 2);


[out.predictions_ols,    ...
 out.predictions_mtl,    ...
 out.predictionVars_mtl, ...
 out.predictionVars_ols]    = deal(cell(n_tasks, 1));
for i = 1:n_tasks

    index_i                 = ((i-1)*(P+1) + 1):(i*(P+1)); 
    Xtest_i                 = designMatTest_cell{i};
    
    if isempty(Xtest_i)
        out.predictions_mtl{i}      = [];
        out.predictionVars_mtl{i}   = [];
        continue;
    end
    
    % OLS predict
    m_ols_i                 = modelOutput.indep.m(index_i);
    out.predictions_ols{i}  = Xtest_i * m_ols_i;
    
    Xtest_i                     = dataTest.designMat_cell{i};
    Xtrain_i                    = modelOutput.dataTrain.designMat_cell{i};
    %out.predictionVars_ols{i}   = var_residuals * Xtest_i * inv(Xtrain_i' * Xtrain_i) * Xtest_i';
    out.predictionVars_ols{i}   = diag(Xtest_i * inv(Xtrain_i' * Xtrain_i) * Xtest_i');
    
    % MTL predict
    m_mtl_i                 = modelOutput.post.m(index_i);
    beta_i                  = modelOutput.post.beta;
    invA_i                  = modelOutput.post.invA(index_i, index_i);
    
    out.predictions_mtl{i}    = designMatTest_cell{i} * m_mtl_i;
    out.predictionVars_mtl{i} = 1/beta_i + diag(Xtest_i*(invA_i * Xtest_i'));
    
end

%out.predVars_ols      	= var_residuals * 

%var_residuals               = ((targets - preds_ols) .* (targets - preds_ols)')/(length(targets) - 2);
%out.predVars_ols            = var_residuals * 


%*********************************************************
function outVec           	= rescale(inVec, stats)

outVec                   	= inVec * stats.sDev + stats.m;

