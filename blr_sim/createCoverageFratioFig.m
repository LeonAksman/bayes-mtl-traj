function createCoverageFratioFig(figureNum, loopVars, metrics_intercept, metrics_slope, model_names)

addpath '../aboxplot';

%***********************************************
% setup stuff

nTasks               = size(loopVars.tasks, 1);
nObservationNoises   = size(loopVars.observationNoises, 2);
nBiomarkerNoises     = size(loopVars.biomarkerNoises, 3);

assert(nBiomarkerNoises == 1);

%close all; 

fontSize            = 25; %15;
fontRotation        = 60;

legendAll           = {};
for i = 1:nObservationNoises
    legendAll{i}    = ['\sigma_m = ', num2str(loopVars.observationNoises(i))];
end

%******************************************
% create the coverage and F ratio figure, which combines intercept and  slope metrics

%*** top row - intercept coupled

%*** add a '*' to indicate oracle models
model_names_int     = model_names;
model_names_int{4} 	= [model_names{4} '*'];
model_names_int{5} 	= [model_names{5} '*'];

figure(figureNum);

subplot(2,3,1);
create_aboxplot(metrics_intercept, 'int',       nObservationNoises, model_names_int, fontRotation, fontSize);
ylim([0 1.1]); 
title('Intercept coverage'); 

subplot(2,3,2);
create_aboxplot(metrics_intercept, 'slope',     nObservationNoises, model_names_int, fontRotation, fontSize);
ylim([0 1.1]); 
title('Slope coverage'); 

subplot(2,3,3);
create_aboxplot(metrics_intercept, 'fRatio',    nObservationNoises, model_names_int, fontRotation, fontSize);
ylim([-3 1.1]); 
title('F stat ratio'); 

 
%*** bottom row - slope coupled

%*** add a '*' to indicate oracle models
model_names_slope    = model_names;
model_names_slope{6} 	= [model_names{6} '*'];
model_names_slope{7} 	= [model_names{7} '*'];

subplot(2,3,4);
create_aboxplot(metrics_slope, 'int',       nObservationNoises, model_names_slope, fontRotation, fontSize);
ylim([0 1.1]); 
title('Intercept coverage'); 

subplot(2,3,5);
create_aboxplot(metrics_slope, 'slope',     nObservationNoises, model_names_slope, fontRotation, fontSize);
ylim([0 1.1]); 
title('Slope coverage'); 

subplot(2,3,6);
create_aboxplot(metrics_slope, 'fRatio',    nObservationNoises, model_names_slope, fontRotation, fontSize);
ylim([-3 1.1]); 
title('F stat ratio'); 
 
leg = legend(legendAll{:});
set(leg, 'Position', [0.8 0.8 0.1 0.1], 'Units', 'normalized'); %set(gca, 'FontSize', 13);
%leg.FontSize = 14;

%saveFigureToPng(1, '~/Desktop/papers/mtl_blr/fig_simulations_coverage_fRatio.png');


%*****************************
function create_aboxplot(m, field, nObservationNoises, model_names, fontRotation, fontSize)

X           = {};
for j = 1:nObservationNoises
    X{j}    = m{1, j}.(field);
end
aboxplot(X, 'labels', model_names, 'colorgrad','green_down', 'plotMean', false); rotateFigFont(fontRotation); set(gca, 'FontSize', fontSize);

