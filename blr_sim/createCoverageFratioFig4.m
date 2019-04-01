function createCoverageFratioFig4(figureNum, loopVars, metrics_intercept, metrics_slope, model_names_int, model_names_slope)

addpath '../aboxplot';

addpath '../subtightplot';
subplot                         = @(m,n,p) subtightplot (m, n, p, [0.17 0.05], [0.15 0.05], [0.04 0.07]);


%***********************************************
% setup stuff

nTasks               = size(loopVars.tasks, 1);
nObservationNoises   = size(loopVars.observationNoises, 2);
nBiomarkerNoises     = size(loopVars.biomarkerNoises, 3);

assert(nBiomarkerNoises == 1);

%close all; 

fontSize            = 20; %25;
fontRotation        = 50; %60;

legendAll           = {};
for i = 1:nObservationNoises
    legendAll{i}    = ['\sigma_m = ', num2str(loopVars.observationNoises(i))];
end

%******************************************
% create the coverage and F ratio figure, which combines intercept and  slope metrics

%*** top row - intercept coupled
figure(figureNum);

subplot(2,4,1);
create_aboxplot(metrics_intercept, 'intCoverage',       nObservationNoises, model_names_int, fontRotation, fontSize);
ylim([0 1.1]); 
title(['Intercept coverage_' char(20)]); 

subplot(2,4,2);
create_aboxplot(metrics_intercept, 'slopeCoverage',     nObservationNoises, model_names_int, fontRotation, fontSize);
ylim([0 1.1]); 
title(['Slope coverage_' char(20)]); 

subplot(2,4,3);
create_aboxplot(metrics_intercept, 'intMae',       nObservationNoises, model_names_int, fontRotation, fontSize);
%ylim([0 1.1]); 
title('Intercept log_1_0MAE'); 

subplot(2,4,4);
create_aboxplot(metrics_intercept, 'slopeMae',     nObservationNoises, model_names_int, fontRotation, fontSize);
%ylim([-4 4]); 
title('Slope log_1_0MAE'); 


%*** bottom row - slope coupled
subplot(2,4,5);
create_aboxplot(metrics_slope, 'intCoverage',       nObservationNoises, model_names_slope, fontRotation, fontSize);
ylim([0 1.1]); 
title(['Intercept coverage_' char(20)]); 

subplot(2,4,6);
create_aboxplot(metrics_slope, 'slopeCoverage',     nObservationNoises, model_names_slope, fontRotation, fontSize);
ylim([0 1.1]); 
title(['Slope coverage_' char(20)]); 

subplot(2,4,7);
create_aboxplot(metrics_slope, 'intMae',       nObservationNoises, model_names_slope, fontRotation, fontSize);
%ylim([0 1.1]); 
title('Intercept log_1_0MAE'); 

subplot(2,4,8);
create_aboxplot(metrics_slope, 'slopeMae',     nObservationNoises, model_names_slope, fontRotation, fontSize);
%ylim([-4 4]);
title('Slope log_1_0MAE'); 

leg = legend(legendAll{:});
set(leg, 'Position', [0.8 0.8 0.1 0.1], 'Units', 'normalized');


% %*****************************************************************
figure(figureNum + 1);
 
subplot                         = @(m,n,p) subtightplot (m, n, p, [0.17 0.10], [0.15 0.07], [0.10 0.10]);

subplot(2,1,1);
create_aboxplot(metrics_intercept, 'predCoverage',       nObservationNoises, model_names_int, fontRotation, fontSize);
ylim([0 1.1]); 
title(['Prediction coverage_' char(20)]); 
%ylim([0.8 1.05]);

subplot(2,1,2);
create_aboxplot(metrics_slope, 'predCoverage',       nObservationNoises, model_names_slope, fontRotation, fontSize);
ylim([0 1.1]); 
title(['Prediction coverage_' char(20)]); 
%ylim([0.8 1.05]);
% leg = legend(legendAll{:});
% set(leg, 'Position', [0.8 0.8 0.1 0.1], 'Units', 'normalized'); 


%saveFigureToPng(1, '~/Desktop/papers/mtl_blr/fig_simulations_coverage_fRatio.png');


%*****************************
function create_aboxplot(m, field, nObservationNoises, model_names, fontRotation, fontSize)

X           = {};
for j = 1:nObservationNoises
    X{j}    = m{1, j}.(field);
end
aboxplot(X, 'labels', model_names, 'colorgrad','green_down', 'plotMean', false, 'plotOutliers', false); 
rotateFigFont(fontRotation); set(gca, 'FontSize', fontSize);

