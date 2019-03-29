function [dataTrain_out, dataTest_out, normStats]            = normalizeDesignMatrices(dataTrain, dataTest)
    
designMatTrain                  = vertcat(dataTrain.designMat{:});
designMatTest                   = vertcat(dataTest.designMat{:});

[normStats.mean, normStats.std] = deal(mean(designMatTrain), std(designMatTrain));

assert(unique(designMatTrain(:, 1)) == 1);
for i = 2:(P+1)
    designMatTrain(:, i)        = (designMatTrain(:, i) - normStats.mean(i)) ./ normStats.std(i);
    designMatTest(:, i)         =  (designMatTest(:, i) - normStats.mean(i)) ./ normStats.std(i); 
end

[normedDesignMatTrain_cell, ...
 normedDesignMatTest_cell]      = deal(cell(n_tasks, 1));

[iTrain, iTest]                 = deal(1);
for i = 1:n_tasks
    normedDesignMatTrain_cell{i}      = designMatTrain((iTrain + (1:nSamples_train(i)) - 1), :);
    normedDesignMatTest_cell{i}       = designMatTest(  (iTest +  (1:nSamples_test(i)) - 1), :);

    iTrain                      = iTrain + nSamples_train(i);
    iTest                       = iTest + nSamples_test(i);
end

dataTrain_out                   = dataTrain;
dataTest_out                    = dataTest;
dataTrain_out.designMat       	= normedDesignMatTrain_cell;
dataTest_out.designMat         	= normedDesignMatTest_cell;
