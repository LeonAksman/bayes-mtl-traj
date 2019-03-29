function [modelOutput, predictStruct, evalOutput] = runAnalysis(rawData, model, indicesCell_train, indicesCell_test)

%Inputs: 
%   data:   structure with fields: subj_id, age, measurement1,...,measurementD
%   params: structure with fields: P (model order)
%

params                              = model.params;

measurements                        = setdiff(fieldnames(rawData), {'subj_id', 'unique_subj_id', 'age', 'age_raw'});

[modelOutput,      ...
 predictStruct,    ...
 evalOutput]                        = deal(struct);

for i = 1:length(measurements)
    
    ids                             = rawData.subj_id;
    
    times_i                         = rawData.age;
    measures_i                      = rawData.(measurements{i});
    
    dataTrain                       = prepData(ids, times_i, measures_i, params.P, indicesCell_train);
    dataTest                        = prepData(ids, times_i, measures_i, params.P, indicesCell_test);
    [dataTrain.name, dataTest.name]  = deal(measurements{i});
    
    targetsTest                     = dataTest.targets_cell;
    
 	%being a bit pedantic: make sure test targets are withheld predict_mtl
    %function, though it doesn't use them
    %clear dataTest.targets_cell;
    
    numTestingSubjects              = sum(~cellfun(@isempty, dataTest.targets_cell));
    dispf('%s, %s: training subjects %d, testing subjects %d', model.name, measurements{i}, dataTrain.n_tasks, numTestingSubjects);
    
    modelOutput.(measurements{i})  	= feval(params.f_train,  dataTrain,                       params);
    predictStruct.(measurements{i})	= feval(params.f_predict,dataTest,                        modelOutput.(measurements{i}));
    evalOutput.(measurements{i})  	= feval(params.f_eval,   predictStruct.(measurements{i}), targetsTest);

end


