function [modelOutput, modelData] = predict_measurement(rawData, model, measurementName, testPoints)


addpath '../utils';
%addpath(genpath('../PRoNTo_v.1.1_r740/machines/gpml/gpml-v3.1'));
addpath(genpath('../gpml-matlab-v4.0-2016-10-19/util'));


if nargin < 4 
    testPoints                  = [];
end

params                          = model.params;

P                               = params.P;
mode                            = params.mode;

if isfield(rawData, 'unique_subj_id')
    unique_subj_ids             = rawData.unique_subj_id;
    rawData                     = rmfield(rawData,  'unique_subj_id');
else
    unique_subj_ids         	= unique(rawData.subj_id);
end
n_tasks                         = length(unique_subj_ids);


%IMPORTANT: make sure subject id order is exactly the same as the one used to create the similarity kernel
assert(isequal(unique_subj_ids, params.kernelSubjectIds));

[trainTimes_cell, testTimes_cell]           = deal(cell(n_tasks, 1));
[targetsTrain_cell, designMatTrain_cell] 	= deal(cell(n_tasks, 1));
[targetsTest_cell, designMatTest_cell]      = deal(cell(n_tasks, 1));
isEmpty                                     = zeros(n_tasks, 1);             %deal with empty test data (subject only used for training)

[nSamples_train, nSamples_test]             = deal(zeros(n_tasks, 1));

for i = 1:n_tasks
    
    if iscell(unique_subj_ids)
        index_i                	= find(ismember(rawData.subj_id, unique_subj_ids{i}));
    else
        index_i              	= find(ismember(rawData.subj_id, unique_subj_ids(i)));
    end
    
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
        
        	nSamples_train(i) 	= length(index_train_i);
            nSamples_test(i)  	= length(index_test_i);
            
        case 'fixed_training'
            assert(params.minTrainingSamples <= n_i);
            
            index_train_i     	= 1:params.minTrainingSamples;
            index_test_i     	= (index_train_i(end) + 1):n_i; %empty if params.minTrainingStamples equal n_i
                
        
        	nSamples_train(i) 	= length(index_train_i);
            nSamples_test(i)  	= length(index_test_i);
            
        case 'predict_multiple'
            assert((params.numTrainingSamples + params.numTestingSamples) == n_i);
            index_train_i     	= 1:params.numTrainingSamples;
            index_test_i        = (params.numTrainingSamples + 1):n_i;
        
        	nSamples_train(i)  	= length(index_train_i);
            nSamples_test(i)   	= length(index_test_i);
    
      	%*********** FOR TADPOLE
        case 'fixed_testing'
            assert(~isempty(testPoints));
            
            index_train_i       = 1:n_i;
            index_test_i      	= [];
            
                    
        	nSamples_train(i) 	= length(index_train_i);
            nSamples_test(i)  	= length(testPoints{i});
            
        otherwise
            error('unknown mode %s', mode);
    end
  
    measurements_i            	= data_i.(measurementName);
    
    %************* TRAINING
    
    t_i_train                	= data_i.age(index_train_i);    
    Z_i_train                	= zeros(nSamples_train(i), P + 1);
    for j = 0:P
        Z_i_train(:, j+1)     	= t_i_train .^ j;
    end
    targetsTrain_i            	= measurements_i(index_train_i);
    
    
    %************* TESTING
    if strcmp(mode, 'fixed_testing')
        
        t_i_test             	= testPoints{i}; 
        Z_i_test              	= zeros(nSamples_test(i), P + 1);
        for j = 0:P
            Z_i_test(:, j+1) 	= t_i_test .^ j;
        end   
        targetsTest_i        	= zeros(length(t_i_test), 1); %measurements_i(index_test_i);
        
        isEmpty(i)              = false; 
     
    elseif ~isempty(index_test_i)
        
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
    trainTimes_cell{i}          = t_i_train;
    testTimes_cell{i}           = t_i_test;
    
    designMatTrain_cell{i}   	= Z_i_train; 
    designMatTest_cell{i}      	= Z_i_test;
    
    targetsTrain_cell{i}      	= targetsTrain_i;        
 	targetsTest_cell{i}         = targetsTest_i;

end

%***********************************
normalizeDesignMatrix               = true; % false; %more stable and higher model evidence when you normalize the design matrix

%normalize the design matrix
if normalizeDesignMatrix
    designMatTrain                  = vertcat(designMatTrain_cell{:});
    designMatTest                   = vertcat(designMatTest_cell{:});
    [designMat_mean, designMat_std] = deal(mean(designMatTrain), std(designMatTrain));
    assert(unique(designMatTrain(:, 1)) == 1);
    for i = 2:(P+1)
        designMatTrain(:, i)        = (designMatTrain(:, i) - designMat_mean(i)) ./ designMat_std(i);
        designMatTest(:, i)         =  (designMatTest(:, i) - designMat_mean(i)) ./ designMat_std(i); 
    end
    [iTrain, iTest]                 = deal(1);
    for i = 1:n_tasks
        designMatTrain_cell{i}      = designMatTrain((iTrain + (1:nSamples_train(i)) - 1), :);
        designMatTest_cell{i}       = designMatTest(  (iTest +  (1:nSamples_test(i)) - 1), :);

        iTrain                      = iTrain + nSamples_train(i);
        iTest                       = iTest + nSamples_test(i);
    end
    modelData.designMat_mean        = designMat_mean;
    modelData.designMat_std         = designMat_std;
end
%***********************************

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
    case 'blr_mtl_mkl_inv_reindex'
        nHyp                    = 2 + (nHyp-1) * (P + 1) ;   %same as above + alpha_diag          
    case 'blr_mtl_mkl_inv_noDiag'
        nHyp                    = 2 + (nHyp-1) * (P + 1) ;   %same as above + alpha_diag           
     case 'blr_mtl_mkl_inv_split'
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
modelData.trainTimes_cell       = trainTimes_cell;
modelData.testTimes_cell        = testTimes_cell; 
modelData.designMat_cell       	= designMatTrain_cell;
modelData.designMatTest_cell  	= designMatTest_cell;
modelData.isEmpty               = isEmpty;

modelOutput                 	= trainTestModel(modelData, 1, params); 
modelOutput.measureName      	= measurementName;

modelOutput.normalizeDesignMatrix = normalizeDesignMatrix;
%***********************************
%rescale the design matrix after modelling
if normalizeDesignMatrix
    
    %do this before rescaling the matrix
    X                             = blkdiag(modelData.designMat_cell{:});
    invA_ols                      = inv(X'*X);
    modelOutput.indep.twoStd_pos  = modelOutput.indep.m + 2 * sqrt(diag(invA_ols));
    modelOutput.indep.twoStd_neg  = modelOutput.indep.m - 2 * sqrt(diag(invA_ols));
    
    [iTrain, iTest]            	= deal(1);
    for i = 1:n_tasks
        train_i               	= modelData.designMat_cell{i};
        test_i                	= modelData.designMatTest_cell{i}; 
        for j = 2:(P+1)
            train_i(:, j)     	= (train_i(:, j) * designMat_std(j)) + designMat_mean(j);
            test_i(:, j)       	=  (test_i(:, j) * designMat_std(j)) + designMat_mean(j);
        end
        modelData.designMat_cell{i}    	= train_i;
        modelData.designMatTest_cell{i}	= test_i;
        iTrain                	= iTrain + nSamples_train(i);
        iTest                 	= iTest + nSamples_test(i);
    end
    modelOutput.designMat_cell      = modelData.designMat_cell;
    modelOutput.designMatTest_cell  = modelData.designMatTest_cell;
    
	modelOutput.designMat_mean    	= designMat_mean;
    modelOutput.designMat_std     	= designMat_std;
    %*** rescale coeffs
    % in normalized variables: y = a * x1_norm + b * x2_norm + c  
    % in original   variables: y = a/sigma1 * x1 + b/sigma2 * x2 + [c - a*mu1/sigma1 - b*mu2/sigma2]
    % ... can be generalized to n variables trivially
    %
    
    invA                                    = modelOutput.post.invA;
    if size(invA, 2) ~= 1
        invA                                = diag(invA);
    end
   	%modelOutput.post.twoStd_pos             = modelOutput.post.m + 2 * sqrt(diag(modelOutput.post.invA));
    %modelOutput.post.twoStd_neg             = modelOutput.post.m - 2 * sqrt(diag(modelOutput.post.invA));
   	modelOutput.post.twoStd_pos             = modelOutput.post.m + 2 * sqrt(invA);
    modelOutput.post.twoStd_neg             = modelOutput.post.m - 2 * sqrt(invA);

    
    %NOTE: to keep it simple, just keeping the diagnoal part of parameter uncertainty
    %modelOutput.post.invA                   = diag(modelOutput.post.invA);
    
    for i = 2:(P+1)
        int_update_i                        = (modelOutput.post.m(i:(P+1):end) * designMat_mean(i))/designMat_std(i);
        %update intercepts
     	modelOutput.post.m(1:(P+1):end)             = modelOutput.post.m(1:(P+1):end)           - int_update_i;  
        modelOutput.post.twoStd_pos(1:(P+1):end)    = modelOutput.post.twoStd_pos(1:(P+1):end)  - int_update_i;
        modelOutput.post.twoStd_neg(1:(P+1):end)    = modelOutput.post.twoStd_neg(1:(P+1):end)  - int_update_i;
           
        %update current coeffs
        modelOutput.post.m(i:(P+1):end)             = modelOutput.post.m(i:(P+1):end)         	/ designMat_std(i);
      	modelOutput.post.twoStd_pos(i:(P+1):end)  	= modelOutput.post.twoStd_pos(i:(P+1):end)	/ designMat_std(i);
     	modelOutput.post.twoStd_neg(i:(P+1):end)  	= modelOutput.post.twoStd_neg(i:(P+1):end)	/ designMat_std(i);
        
        modelOutput.post.invA(i:(P+1):end)          = modelOutput.post.invA(i:(P+1):end)/(designMat_std(i)^ 2); %alternative: *
        
        
        %****************** indep
    	%update intercepts
        int_update_i_indep                       	= (modelOutput.indep.m(i:(P+1):end) * designMat_mean(i))/designMat_std(i);
     	modelOutput.indep.m(1:(P+1):end)          	= modelOutput.indep.m(1:(P+1):end) - int_update_i_indep;
        modelOutput.indep.twoStd_pos(1:(P+1):end)   = modelOutput.indep.twoStd_pos(1:(P+1):end) - int_update_i_indep;
        modelOutput.indep.twoStd_neg(1:(P+1):end)   = modelOutput.indep.twoStd_neg(1:(P+1):end) - int_update_i_indep;
          
        %update current coeffs
        modelOutput.indep.m(i:(P+1):end)             = modelOutput.indep.m(i:(P+1):end)             / designMat_std(i);
      	modelOutput.indep.twoStd_pos(i:(P+1):end)    = modelOutput.indep.twoStd_pos(i:(P+1):end)	/ designMat_std(i);
     	modelOutput.indep.twoStd_neg(i:(P+1):end)  	 = modelOutput.indep.twoStd_neg(i:(P+1):end)	/ designMat_std(i);
           
    end
end
%***********************************


alphas                          = modelOutput.hyp(2:end); %exp(modelOutput.hyp(2:end));
dispf('beta: %.1f, alpha diag: %f, other alphas: %s', modelOutput.hyp(1), alphas(1), vecToString('%.05f  ', alphas(2:end)));

%*********************************************************
function [trainOut, testOut]  	= normalize_local(trainIn, testIn, nSamples_train, nSamples_test)

trainVec                        = vertcat(trainIn{:});
testVec                         = vertcat(testIn{:});

[m, stdDev]                   	= deal(mean(trainVec), std(trainVec));
trainVec                        = (trainVec - m)/stdDev;
testVec                         = (testVec - m)/stdDev;

trainOut                        = toTasksCell(trainVec, nSamples_train);
testOut                         = toTasksCell(testVec,  nSamples_test);
