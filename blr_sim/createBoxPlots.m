function createBoxPlots(loadFilename)

load(loadFilename); %loads metrics_all, model_names

nTasks               = size(metrics_all, 1);
nObservationNoises   = size(metrics_all, 2);
nBiomarkerNoises     = size(metrics_all, 3);


tasks                        	= 50;      %[20 50 100 150 200 250 300]; 
observationNoises            	= [0.1 1 2 4 8];
biomarkerNoises             	= [0.1 1 2 4 8];

close all;

fontSize            = 15;
fontRotation        = 30;

for i = 1:nTasks
    for j = 1:nObservationNoises
        for k = 1:nBiomarkerNoises
        
            metrics_ij                      = metrics_all{i, j, k};      
            
            subplot(2,2,1); boxplot(metrics_ij.int,     'Labels', model_names); set(gca, 'FontSize', fontSize); rotateFig(fontRotation); title('Intercept coverage');	
            subplot(2,2,2);	boxplot(metrics_ij.slope,   'Labels', model_names); set(gca, 'FontSize', fontSize); rotateFig(fontRotation); title('Slope coverage');	
            subplot(2,2,3);	boxplot(metrics_ij.mae,     'Labels', model_names); set(gca, 'FontSize', fontSize); rotateFig(fontRotation); title('MAE predictions');	
            subplot(2,2,4);	boxplot(metrics_ij.fRatio, 	'Labels', model_names); set(gca, 'FontSize', fontSize); rotateFig(fontRotation); ylim([0 1]); title('F ratio');	
            
            dispf('tasks: %d, obs noise: %.1f, bio noise: %.1f', tasks(i), observationNoises(j), biomarkerNoises(k));
            pause;
            %if i > 1
            %    pause
            %end
            
         	%[~, h1]     = suplabel('blah', 't');
            %set(h1,'FontSize',30)
        end
    end
end
