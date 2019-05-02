function out                = predict_mtl(dataTest, modelOutput) 

out.dataTest                = dataTest;

n_tasks                     = dataTest.n_tasks;
P                           = modelOutput.params.P;

%** new
targetsTrain_cell           = modelOutput.dataTrain.targets_cell;
designMatTrain_cell         = normalizeDesignMat(modelOutput.dataTrain.designMat_cell, P, modelOutput.stats_design);
designMatTest_cell       	= normalizeDesignMat(dataTest.designMat_cell,              P, modelOutput.stats_design);
%designMatTest_cell          = dataTest.designMat_cell;

assert(length(designMatTest_cell) == n_tasks);
assert(length(modelOutput.post.beta) == 1);

[out.predictions_ols,    ...
 out.predictions_mtl,    ...
 out.predictionVars_mtl, ...
 out.predictionVars_ols]    = deal(cell(n_tasks, 1));
for i = 1:n_tasks

    index_i                 = ((i-1)*(P+1) + 1):(i*(P+1)); 
    Xtest_i                 = designMatTest_cell{i};
    
    ytrain_i                = targetsTrain_cell{i};
    Xtrain_i             	= designMatTrain_cell{i};
    
    if isempty(Xtest_i)
        out.predictions_mtl{i}      = [];
        out.predictionVars_mtl{i}   = [];
        continue;
    end
    
    % ***************** OLS predict
    %*** new
    %m_ols_i                 = modelOutput.indep.m(index_i);
    %m_ols_i                 = modelOutput.indep_unadjusted.m(index_i);
    m_ols_i                 = pinv(Xtrain_i) * ytrain_i;
    out.predictions_ols{i}  = Xtest_i * m_ols_i;   
    out.predictionVars_ols{i}   = diag(Xtest_i * inv(Xtrain_i' * Xtrain_i) * Xtest_i');

    %*** new
    %out.predictions_ols{i}      = rescale(out.predictions_ols{i}, modelOutput.stats_targets);
  	%out.predictionVars_ols{i}   = out.predictionVars_ols{i} * (modelOutput.stats_targets.sDev .^ 2);     
    
    % ***************** MTL predict
    %m_mtl_i                 = modelOutput.post.m(index_i);
    %beta_i                  = modelOutput.post.beta;
    %invA_i                  = modelOutput.post.invA(index_i, index_i);
    m_mtl_i                 = modelOutput.post_unadjusted.m(index_i);
    beta_i                  = modelOutput.post_unadjusted.beta;
    invA_i                  = modelOutput.post_unadjusted.invA(index_i, index_i);
    
    out.predictions_mtl{i}    = Xtest_i * m_mtl_i;
    out.predictionVars_mtl{i} = 1/beta_i + diag(Xtest_i*(invA_i * Xtest_i'));
    
    %*** new
  	out.predictions_mtl{i}      = rescale(out.predictions_mtl{i}, modelOutput.stats_targets);
  	out.predictionVars_mtl{i}   = out.predictionVars_mtl{i} * (modelOutput.stats_targets.sDev .^ 2);
end

%*********************************************************
function outVec           	= rescale(inVec, stats)

outVec                   	= inVec * stats.sDev + stats.m;

