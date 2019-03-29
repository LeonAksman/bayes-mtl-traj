function out                = predict_matlabLME(dataTest, modelOutput) 

out.dataTest                = dataTest;

params                      = modelOutput.params;

n_tasks                     = dataTest.n_tasks;
designMatTest_cell          = dataTest.designMat_cell;

assert(length(designMatTest_cell) == n_tasks);

P                           = modelOutput.params.P;


out.predictions_lme         = cell(n_tasks, 1);

for i = 1:n_tasks

    targets_i               	= dataTest.targets_cell{i};
    
    if isempty(targets_i)
        out.predictions_lme{i}   = [];
        continue;
    end
    
    n_i                     = length(targets_i);
    
    %Xtest_i                 = [Xtest_i repmat(params.fixedEffects(i, :), [dataTest.nSamples(i) 1])];
    %out.predictions_lme{i}   = Xtest_i * modelOutput.post.b(i, :)';

    tbl_i                     = table;
    tbl_i.(dataTest.name)     = targets_i;
    tbl_i.id                  = repVec(dataTest.ids(i), n_i);
    tbl_i.t                   = vertcat(dataTest.times_cell{i});
    for j = 1:size(params.X_vals, 2)
        tbl_i.(params.X_names{j}) = repVec(modelOutput.params.X_vals(i, j), n_i); %dataTest.nSamples);
    end  
    
    out.predictions_lme{i}   	= predict(modelOutput.post.lme, tbl_i);
    
end


