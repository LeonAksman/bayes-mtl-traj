function modelOutputs           	= predict_blr_mtl(rawData, model)
%Inputs: 
%   data:   structure with fields: subj_id, age, measurement1,...,measurementD
%   params: structure with fields: P (model order), mode ('predict_last')
%

addpath '../utils';
addpath(genpath('../PRoNTo_v.1.1_r740/machines/gpml/gpml-v3.1'));

params                          = model.params;
measurements                    = setdiff(fieldnames(rawData), {'subj_id', 'age', 'age_raw'});

modelOutputs                    = [];
for i = 1:length(measurements)
    [modelOutput_i, modelData_i] = predict_measurement(rawData, params, measurements{i});
    modelOutput_i.modelName     = model.name;
    
    modelOutputs                = [modelOutputs; modelOutput_i];
    
    %if strcmp(params.plotMeasure, measurements{i})
    %    plot_mtl(modelOutput_i, params.figureNum);
    %end
end



%******************************************
function [modelOutput, modelData] = predict_measurement(rawData, params, measurementName)

P                               = params.P;
mode                            = params.mode;
assert(strcmp(mode, 'predict_last'));

unique_subj_ids                 = unique(rawData.subj_id);
n_tasks                         = length(unique_subj_ids);

[targetsTrain_cell, designMatTrain_cell] 	= deal(cell(n_tasks, 1));
[targetsTest_cell, designMatTest_cell]      = deal(cell(n_tasks, 1));

[nSamples_train, nSamples_test]             = deal(zeros(n_tasks, 1));

for i = 1:n_tasks
    
    index_i                     = find(ismember(rawData.subj_id, unique_subj_ids{i}));
    data_i                      = reindexStruct(rawData, index_i);
    
    %*************** use raw age for re-indexing, but later predict age (which may be brain age or raw age)
    [~, index_i2]             	= sort(data_i.age_raw, 'ascend');
    data_i                      = reindexStruct(data_i, index_i2);

    n_i                         = length(index_i);
    
    error('fix me from here');
    switch mode
        case 'predict_last'
            index_train_i     	= 1:(n_i - 1);
            index_test_i      	= n_i;
        otherwise
            error('unknown mode %s', mode);
    end
    
    nSamples_train(i)           = length(index_train_i);
    nSamples_test(i)            = length(index_test_i);
    
    measurements_i            	= data_i.(measurementName);
    
    %************* TRAINING
    
    t_i_train                	= data_i.age(index_train_i);    
    Z_i_train                	= zeros(nSamples_train(i), P + 1);
    for j = 0:P
        Z_i_train(:, j+1)     	= t_i_train .^ j;
    end
    targetsTrain_i            	= measurements_i(index_train_i);
    
    
    %************* TESTING
    
    %Instead of predicting brain age, which we do not know in the future as
    %it is a function of the biomarkers we are trying to predict, we
    %instead just add the difference between the true age at prediction
    %and the last available true age in training to the last available brain age in training.
    %In other words we make a crude prediction of brain age at the prediction
    %time-point, then feed this into the model to get a prediction of the
    %biomarker at that time.
    %
    test_ages                 	= data_i.age(index_train_i(end)) + (data_i.age_raw(index_test_i) - data_i.age_raw(index_train_i(end)));
    
    t_i_test                    = test_ages;                        %if using brain age, we don't have this: data_i.age(index_test_i);    
    Z_i_test                    = zeros(nSamples_test(i), P + 1);
    for j = 0:P
        Z_i_test(:, j+1)      	= t_i_test .^ j;
    end   
    targetsTest_i               = measurements_i(index_test_i);
    
    %** assemble
    designMatTrain_cell{i}   	= Z_i_train; 
    designMatTest_cell{i}      	= Z_i_test;
    
    targetsTrain_cell{i}      	= targetsTrain_i;        
 	targetsTest_cell{i}         = targetsTest_i;

end

%*** create necessary matrices

modelData.P                    	= P;
modelData.n_tasks             	= n_tasks;
modelData.nSamples_train      	= nSamples_train;
modelData.nSamples_test       	= nSamples_test;

modelData.extraKernels          = params.extraKernels;

nHyp                            = 3 + numHyp_kernel(params.extraKernels);
switch func2str(params.f_blr)
    case 'blr_mtl_mkl_test'
        nHyp                	= nHyp + n_tasks - 1;
    case 'blr_mtl_mkl_test2'
        nHyp                	= nHyp + P + 1;
end
    
modelData.hyp                   = zeros(nHyp, 1);

modelData.targets_cell         	= targetsTrain_cell;
modelData.targetsTest_cell    	= targetsTest_cell;
modelData.designMat_cell       	= designMatTrain_cell;
modelData.designMatTest_cell  	= designMatTest_cell;


modelOutput                 	= trainTestModel(modelData, 1, params.f_blr);
modelOutput.measureName      	= measurementName;



