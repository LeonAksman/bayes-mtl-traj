function out                = predict_fsLME(dataTest, modelOutput) 

out.dataTest                = dataTest;

params                      = modelOutput.params;

n_tasks                     = dataTest.n_tasks;
designMatTest_cell          = dataTest.designMat_cell;

assert(length(designMatTest_cell) == n_tasks);

P                           = modelOutput.params.P;


out.predictions_lme         = cell(n_tasks, 1);

for i = 1:n_tasks

    Xtest_i                 = dataTest.designMat_cell{i};
    
    if isempty(Xtest_i)
        out.predictions_lme{i}   = [];
        continue;
    end
    
    Xtest_i                 = [Xtest_i repmat(params.fixedEffects(i, :), [dataTest.nSamples(i) 1])];
    out.predictions_lme{i}  = Xtest_i * modelOutput.post.b(i, :)';
    
    
    %similar to : out.predictionVars_mtl{i} = 1/beta_i + diag(Xtest_i*(invA_i * Xtest_i'));
    %using:        SIGMA(posi:posf,1:ni(i)) = Zi*D*Zi'+ eye(ni(i))*phisq;
    n_i                     = dataTest.nSamples(i);
    Z_i                     = Xtest_i(:, params.randomEffectsVec);
    out.predictionVars_lme{i} = diag(Z_i * modelOutput.post.Dhat * Z_i' + eye(n_i) * modelOutput.post.phisqhat);  
end


