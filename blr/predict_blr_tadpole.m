function modelOutputs           	= predict_blr_tadpole(rawData, model)
%Inputs: 
%   data:   structure with fields: subj_id, age, measurement1,...,measurementD
%   params: structure with fields: P (model order), mode ('predict_last')
%

addpath '../utils';
addpath(genpath('../PRoNTo_v.1.1_r740/machines/gpml/gpml-v3.1'));
%addpath(genpath('../gpml-matlab-v4.0-2016-10-19/util'));

measurements                    = setdiff(fieldnames(rawData), {'subj_id', 'age', 'age_raw'});

modelOutputs                    = [];
for i = 1:length(measurements)
    modelOutputs.(measurements{i}) = predict_measurement(rawData, model, measurements{i});
end



%******************************************
function [modelOutput, modelData] = predict_measurement(rawData, model, measurementName)

params                          = model.params;

P                               = params.P;
mode                            = params.mode;

unique_subj_ids                 = unique(rawData.subj_id);
n_tasks                         = length(unique_subj_ids);

[targetsTrain_cell, designMatTrain_cell] 	= deal(cell(n_tasks, 1));
[targetsTest_cell, designMatTest_cell]      = deal(cell(n_tasks, 1));
isEmpty                                     = zeros(n_tasks, 1);             %deal with empty test data (subject only used for training)

[nSamples_train, nSamples_test]             = deal(zeros(n_tasks, 1));

for i = 1:n_tasks
    
    index_i                     = find(ismember(rawData.subj_id, unique_subj_ids{i}));
    data_i                      = reindexStruct(rawData, index_i);
    
    %*************** use raw age for re-indexing, but later predict age (which may be brain age or raw age)
    [~, index_i2]             	= sort(data_i.age_raw, 'ascend');
    data_i                      = reindexStruct(data_i, index_i2);

    n_i                         = length(index_i);
    
    switch mode
                
        case 'predict_last'
            %index_train_i     	= 1:(n_i - 1);
            %index_test_i      	= n_i;
            
            index_train_i     	= 1:max(params.minTrainingSamples, n_i - 1);
            index_test_i     	= (index_train_i(end) + 1):n_i; %empty if params.minTrainingStamples equal n_i

        case 'fixed_training'
            assert(params.minTrainingSamples <= n_i);
            
            index_train_i     	= 1:params.minTrainingSamples;
            index_test_i     	= (index_train_i(end) + 1):n_i; %empty if params.minTrainingStamples equal n_i
                
            
        case 'predict_multiple'
            assert((params.numTrainingSamples + params.numTestingSamples) == n_i);
            index_train_i     	= 1:params.numTrainingSamples;
            index_test_i        = (params.numTrainingSamples + 1):n_i;
        
      	%*********** FOR TADPOLE
        case 'fixed_testing'
            index_train_i       = 1:n_i;
            index_test_i      	= [];
            
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
    if ~isempty(index_test_i)
        
        %Instead of predicting brain age, which we do not know in the future as
        %it is a function of the biomarkers we are trying to predict, we
        %instead just add the difference between the true age at prediction
        %and the last available true age in training to the last available brain age in training.
        %In other words we make a crude prediction of brain age at the prediction
        %time-point, then feed this into the model to get a prediction of the
        %biomarker at that time.       
        test_ages               = data_i.age(index_train_i(end)) + (data_i.age_raw(index_test_i) - data_i.age_raw(index_train_i(end)));

        t_i_test             	= test_ages;                        %if using brain age, we don't have this: data_i.age(index_test_i);    
        Z_i_test              	= zeros(nSamples_test(i), P + 1);
        for j = 0:P
            Z_i_test(:, j+1) 	= t_i_test .^ j;
        end   
        targetsTest_i        	= measurements_i(index_test_i);
        
        isEmpty(i)              = false;   
    elseif strcmp(mode, 'fixed_testing')
        
     	test_ages               = data_i.test_points;

        t_i_test             	= test_ages;                        %if using brain age, we don't have this: data_i.age(index_test_i);    
        Z_i_test              	= zeros(nSamples_test(i), P + 1);
        for j = 0:P
            Z_i_test(:, j+1) 	= t_i_test .^ j;
        end   
        targetsTest_i        	= measurements_i(index_test_i);
        
        isEmpty(i)              = false; 
        
    else          %*** handle case of only training data
       
       t_i_test                 = 0; 
       Z_i_test              	= zeros(1, P + 1);
        for j = 0:P
            Z_i_test(:, j+1) 	= t_i_test .^ j;
        end   
        targetsTest_i        	= 0;

        isEmpty(i)              = true;
    end
    
    %** assemble
    designMatTrain_cell{i}   	= Z_i_train; 
    designMatTest_cell{i}      	= Z_i_test;
    
    targetsTrain_cell{i}      	= targetsTrain_i;        
 	targetsTest_cell{i}         = targetsTest_i;

end

dispf('%s, %s: training samples %d, testing samples %d', model.name, measurementName, n_tasks, sum(~isEmpty));

%*** create necessary matrices

modelData.P                    	= P;
modelData.subj_id             	= unique_subj_ids;
modelData.n_tasks             	= n_tasks;
modelData.nSamples_train      	= nSamples_train;
modelData.nSamples_test       	= nSamples_test;

modelData.extraKernels          = params.extraKernels;

nHyp                            = 3 + numHyp_kernel(params.extraKernels);%beta, alpha1, alpha2 + kernel weights
switch func2str(params.f_blr)
    case 'blr_mtl_mkl_test'
        nHyp                	= nHyp + n_tasks - 1;
    case 'blr_mtl_mkl_test2'
        nHyp                	= nHyp + P + 1;
    case 'blr_mtl_mkl_block'
        nHyp                    = 1 + (nHyp-1) * (P + 1) + 1; %replicate hyperparams for every dimentions (removing beta) + alpha_diag
    case 'blr_mtl_mkl_block_diag'
        nHyp                    = 2 + (nHyp-1) * (P + 1) ;   %same as above + alpha_diag
    case 'blr_mtl_mkl_chol'
        nHyp                    = 2 + (nHyp-1) * (P + 1) ;   %same as above + alpha_diag     
    case 'blr_mtl_mkl_inv'
        nHyp                    = 2 + (nHyp-1) * (P + 1) ;   %same as above + alpha_diag    
    case 'blr_mtl_mkl_inv_fast'
        nHyp                    = 2 + (nHyp-1) * (P + 1) ;   %same as above + alpha_diag            
    case 'blr_mtl_mkl_inv_diag'    
        nHyp                    = 2 + (1 + n_tasks + numHyp_kernel(params.extraKernels)) * (P + 1); %overall: beta, alpha_diag, block: alpha1's, alpha2, kernel related       
    case 'blr_mtl_mkl_block_diag_diag'    
        nHyp                    = 2 + (1 + n_tasks + numHyp_kernel(params.extraKernels)) * (P + 1); %overall: beta, alpha_diag, block: alpha1's, alpha2, kernel related
    case 'blr_mtl_mkl_block_all'
        nHyp                    = 1 + (nHyp-1) * (P + 2); %replicate hyperparams for every dimentions (removing beta) + plus all       
end
 
modelData.hyp                   = zeros(nHyp, 1);

if isfield(params, 'disableOnes')
    if params.disableOnes
        modelData.hyp(3)    	= -Inf;
    end
end

modelData.targets_cell         	= targetsTrain_cell;
modelData.targetsTest_cell    	= targetsTest_cell;
modelData.designMat_cell       	= designMatTrain_cell;
modelData.designMatTest_cell  	= designMatTest_cell;

modelData.isEmpty               = isEmpty;

modelOutput                 	= trainTestModel(modelData, 1, params); 
modelOutput.measureName      	= measurementName;

alphas                          = modelOutput.hyp(2:end); %exp(modelOutput.hyp(2:end));
dispf('alphas: %s', vecToString('%.05f  ', alphas));

%*********************************************************
function [trainOut, testOut]  	= normalize_local(trainIn, testIn, nSamples_train, nSamples_test)

trainVec                        = vertcat(trainIn{:});
testVec                         = vertcat(testIn{:});

[m, stdDev]                   	= deal(mean(trainVec), std(trainVec));
trainVec                        = (trainVec - m)/stdDev;
testVec                         = (testVec - m)/stdDev;

trainOut                        = toTasksCell(trainVec, nSamples_train);
testOut                         = toTasksCell(testVec,  nSamples_test);
