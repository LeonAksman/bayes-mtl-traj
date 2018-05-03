function modelOutputs           	= predict_blr_mtl_flex(rawData, model, testPoints)
%Inputs: 
%   data:   structure with fields: subj_id, age, measurement1,...,measurementD
%   params: structure with fields: P (model order), mode ('predict_last')
%

if nargin < 3
    testPoints                  = [];
end

measurements                    = setdiff(fieldnames(rawData), {'subj_id', 'unique_subj_id', 'age', 'age_raw'});

modelOutputs                    = [];
for i = 1:length(measurements)
    modelOutputs.(measurements{i}) = predict_measurement(rawData, model, measurements{i}, testPoints);
end
