function modelData              = formModelData(rawData, params, measurementName)


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

modelData.targets_cell         	= targetsTrain_cell;
modelData.targetsTest_cell    	= targetsTest_cell;

modelData.designMat_cell       	= designMatTrain_cell;
modelData.designMatTest_cell  	= designMatTest_cell;


