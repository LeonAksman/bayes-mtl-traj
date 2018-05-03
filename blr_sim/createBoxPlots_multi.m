function createBoxPlots_multi(loadFilename)

load(loadFilename); %loads metrics_all, model_names

nTasks               = size(loopVars.tasks, 1);
nObservationNoises   = size(loopVars.observationNoises, 2);
nBiomarkerNoises     = size(loopVars.biomarkerNoises, 3);

% nModels                 = length(model_names);
% nRng                    = size(metrics_all{1, 1, 1}.int, 1);
% tasks                	= 100;      %[20 50 100 150 200 250 300]; 
% observationNoises     	= [0.1 1 2 4 8];
% biomarkerNoises      	= [0.1 1 2 4 8];

close all; 

fontSize            = 15;
fontRotation        = 30;

for i = 1:nTasks
    for j = 1:nObservationNoises
        
        figure(j);
        subplot(2,2,1);
        X           = {};
        for k = 1:nBiomarkerNoises
            X{k}    = metrics_all{i, j, k}.int;
        end
        aboxplot(X, 'labels', model_names); rotateFigFont(fontRotation);
        
     	subplot(2,2,2);
        X           = {};
        for k = 1:nBiomarkerNoises
            X{k}    = metrics_all{i, j, k}.slope;
        end
        aboxplot(X, 'labels', model_names); rotateFigFont(fontRotation);

        subplot(2,2,3);
        X           = {};
        for k = 1:nBiomarkerNoises
            X{k}    = metrics_all{i, j, k}.mae;
        end
        aboxplot(X, 'labels', model_names); ylim([0 10]); rotateFigFont(fontRotation);
        
      	subplot(2,2,4);
        X           = {};
        for k = 1:nBiomarkerNoises
            X{k}    = metrics_all{i, j, k}.fRatio;
        end
        aboxplot(X, 'labels', model_names); ylim([0 1]); rotateFigFont(fontRotation);
        
        dispf('tasks: %d, obs noise: %.1f', loopVars.tasks(i), loopVars.observationNoises(j)); 
        %pause;
        %X           = zeros(nRng, nModels*nBiomarkerNoises);
        
        %for k = 1:nModels
        %    X(:, k:nModels:end) = metrics_all{i, j, k}.int;
        %end
        
        %bio1 = metrics_all{i, j, 1}.int;
        
%         for k = 1:nBiomarkerNoises
%         
%             metrics_ij                      = metrics_all{i, j, k};      
%             
%             subplot(2,2,1); boxplot(metrics_ij.int,     'Labels', model_names); set(gca, 'FontSize', fontSize); rotateFig(fontRotation); title('Intercept coverage');	
%             subplot(2,2,2);	boxplot(metrics_ij.slope,   'Labels', model_names); set(gca, 'FontSize', fontSize); rotateFig(fontRotation); title('Slope coverage');	
%             subplot(2,2,3);	boxplot(metrics_ij.mae,     'Labels', model_names); set(gca, 'FontSize', fontSize); rotateFig(fontRotation); title('MAE predictions');	
%             subplot(2,2,4);	boxplot(metrics_ij.fRatio, 	'Labels', model_names); set(gca, 'FontSize', fontSize); rotateFig(fontRotation); ylim([0 1]); title('F ratio');	
%             
%             dispf('tasks: %d, obs noise: %.1f, bio noise: %.1f', tasks(i), observationNoises(j), biomarkerNoises(k));
%             pause;
%             %if i > 1
%             %    pause
%             %end
%             
%          	%[~, h1]     = suplabel('blah', 't');
%             %set(h1,'FontSize',30)
%         end
    end
end
