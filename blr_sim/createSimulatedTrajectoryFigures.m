function createSimulatedTrajectoryFigures()

addpath '../tightfigadv';

close all;

params.n_tasks                  = 200;
params.rngSeed                  = 1;

observationNoises               = [1 2 4 8];
for i = 1:length(observationNoises)
    
    params.observationNoise     = observationNoises(i);       
    
    %*** intercept
    [predictStruct_i, coeffs_i]	= generatePredictionStructure_intercepts(params);
   
    group_i                     = toGroupNumber(coeffs_i(:, 1));
    
    subplot(2, 4, i);
    plot_local(predictStruct_i, group_i);
    title(['\sigma_m = ', num2str(observationNoises(i))]); %, 'FontSize', 15);
    set(gca, 'FontSize', 25);
    ylim([-40 20]); 
    yticks([-40 0 20]);
    xticks([0 5 10]);
    
    
    
    %*** slope
    [predictStruct_i, coeffs_i] = generatePredictionStructure_slopes(params);
        
    group_i                     = toGroupNumber(coeffs_i(:, 2));
    
    subplot(2, 4, 4+i);
    plot_local(predictStruct_i, group_i);    
    title(['\sigma_m = ', num2str(observationNoises(i))]); %, 'FontSize', 15);
    set(gca, 'FontSize', 25);
    ylim([-40 20]); 
    yticks([-40 0 20]);
    xticks([0 5 10]);
end

tightfigadv;

%*************************
function plot_local(predStruct, group)

uniqueIds                   = unique(predStruct.subj_id);
nTasks                      = length(uniqueIds);

%colorCodes                  = {'y', 'm', 'c', 'r', 'g', 'b'};
colorCodes                  = {'m', 'r', 'g', 'b', 'k'};

hold on;

for i = 1:nTasks
    
    index_i                 = find(predStruct.subj_id == uniqueIds(i));
    
    vals_i                  = predStruct.sim(index_i);
    t_i                     = predStruct.age(index_i);
       
    %color_i                 = randsample(colorCodes, 1);
    
    plot(t_i, vals_i, sprintf('-%s.', colorCodes{group(i)})); %color_i{1}));  
end

hold off;

%******************
function groupNum   = toGroupNumber(coeffs)
    
uniqueVals          = unique(coeffs);
nUniqueVals         = length(uniqueVals);

groupNum            = zeros(size(coeffs));
for i = 1:nUniqueVals
    index_i             = find(coeffs == uniqueVals(i));
    groupNum(index_i)   = i;
end

