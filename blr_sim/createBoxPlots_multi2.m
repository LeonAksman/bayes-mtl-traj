function createBoxPlots_multi2(loadFilename)

load(loadFilename); %loads metrics_all, model_names

%**remove first observation noise column
metrics_all                 = metrics_all(:, 2:end, :);
loopVars.observationNoises  = loopVars.observationNoises(2:end);

nTasks               = size(loopVars.tasks, 1);
nObservationNoises   = size(loopVars.observationNoises, 2);
nBiomarkerNoises     = size(loopVars.biomarkerNoises, 3);

assert(nBiomarkerNoises == 1);


% nModels                 = length(model_names);
% nRng                    = size(metrics_all{1, 1, 1}.int, 1);
% tasks                	= 100;      %[20 50 100 150 200 250 300]; 
% observationNoises     	= [0.1 1 2 4 8];
% biomarkerNoises      	= [0.1 1 2 4 8];

close all; 

fontSize            = 15;
fontRotation        = 30;

legendAll           = {};
for i = 1:nObservationNoises
    legendAll{i}    = ['\sigma_m = ', num2str(loopVars.observationNoises(i))];
end

for i = 1:nTasks
    subplot(2,2,1);
    X           = {};
    for j = 1:nObservationNoises
        X{j}    = metrics_all{i, j, 1}.int;
    end
    aboxplot(X, 'labels', model_names, 'plotMean', false); rotateFigFont(fontRotation); set(gca, 'FontSize', fontSize);

    subplot(2,2,2);
    X           = {};
    for j = 1:nObservationNoises
        X{j}    = metrics_all{i, j, 1}.slope;
    end
    aboxplot(X, 'labels', model_names, 'plotMean', false); rotateFigFont(fontRotation); set(gca, 'FontSize', fontSize);

    subplot(2,2,3);
    X           = {};
    for j = 1:nObservationNoises
        X{j}    = metrics_all{i, j, 1}.mae;
    end
    aboxplot(X, 'labels', model_names, 'plotMean', false); ylim([0 10]); rotateFigFont(fontRotation);  set(gca, 'FontSize', fontSize);

    subplot(2,2,4);
    X           = {};
    for j = 1:nObservationNoises
        X{j}    = metrics_all{i, j, 1}.fRatio;
    end
    aboxplot(X, 'labels', model_names, 'plotMean', false); ylim([0 1]); rotateFigFont(fontRotation); set(gca, 'FontSize', fontSize);

    leg1 = legend(legendAll{:});
    set(leg1, 'Position', [0.4 0.4 0.2 0.2], 'Units', 'normalized'); set(gca, 'FontSize', 10);
end
