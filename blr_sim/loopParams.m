function [metrics_all, model_names, loopVars] = loopParams(loopVars, f_generatePredictionStructure, style, saveFilename)

if nargin < 4
     saveFilename                       = '';
end

if ~isempty(saveFilename) && fileExist(saveFilename)
    strInput = input(sprintf('File %s exists. Do you want to delete this file and rerun? (y/n) ', saveFilename), 's');
    if lower(strInput(1)) ~= 'y'
        
     	load(saveFilename);
        disp('Loaded variables in save file. Exiting');
        
        return;
    else
        dispf('Continuing, overwritting %s', saveFilename);
    end
end

metrics_all                             = {};

for i = 1:length(loopVars.tasks)
    for j = 1:length(loopVars.observationNoises)
        for k = 1:length(loopVars.biomarkerNoises)
            [metrics.predCoverage,      ...
             metrics.intCoverage,     	...   
             metrics.slopeCoverage,   	... 
             metrics.mae,               ...
             metrics.intMae,          	...
             metrics.slopeMae,        	...
             metrics.fRatio]                   = deal([]);

            for kk = 1:loopVars.numRngs
                inParams_i.n_tasks              = loopVars.tasks(i); 
                inParams_i.observationNoise     = loopVars.observationNoises(j);
                inParams_i.biomarkerNoise      	= loopVars.biomarkerNoises(k);
                inParams_i.rngSeed              = kk;
                inParams_i.numTestSamples       = loopVars.numTestSamples;
                
                dispf('*********** tasks: %d, noise: %d, seed; %d', loopVars.tasks(i), loopVars.observationNoises(j), kk);
                %try
              	[metrics_i, model_names]  	= computeModelMetrics(inParams_i, f_generatePredictionStructure, style);
                %catch
                %    dispf('Caught error: %s ********************', lasterr);
                %    continue;
                %end

                metrics.predCoverage        = [metrics.predCoverage;    metrics_i.predCoverage];
                metrics.intCoverage       	= [metrics.intCoverage;   	metrics_i.intCoverage];
                metrics.slopeCoverage      	= [metrics.slopeCoverage; 	metrics_i.slopeCoverage];
                metrics.mae             	= [metrics.mae;             metrics_i.mae];
                metrics.intMae              = [metrics.intMae;       	metrics_i.intMae];
                metrics.slopeMae            = [metrics.slopeMae;      	metrics_i.slopeMae];
                metrics.fRatio             	= [metrics.fRatio;          metrics_i.fRatio];


            end

            metrics_all{i, j, k}           	= metrics;
        end
    end
end

if ~isempty(saveFilename)
    save(saveFilename, 'metrics_all', 'model_names', 'loopVars');
end

% close all;
% for i = 1:size(mean_ints,2)
%     figure(1); plot(squeeze(mean_ints(1, i, :)));   ylim([0 1]);
%     figure(2); plot(squeeze(mean_slopes(1, i, :))); ylim([0 1]);
%     figure(3); plot(squeeze(mean_maes(1, i, :)));
%     figure(4); plot(squeeze(median_ratio(1, i, :)));
%     disp(i)
%     if i > 1
%         pause
%     end
% end