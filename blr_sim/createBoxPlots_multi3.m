function createBoxPlots_multi3(loadFilename_intercepts, loadFilename_slopes)

addpath '..\aboxplot';

load(loadFilename_intercepts);
metrics_intercept         	= metrics_all; %(:, 2:end, :);

load(loadFilename_slopes); 
metrics_slope           	= metrics_all; %(:, 2:end, :);

%loopVars.observationNoises  = loopVars.observationNoises(2:end);

%***********************************************
% setup stuff

nTasks               = size(loopVars.tasks, 1);
nObservationNoises   = size(loopVars.observationNoises, 2);
nBiomarkerNoises     = size(loopVars.biomarkerNoises, 3);

assert(nBiomarkerNoises == 1);

close all; 

fontSize            = 15;
fontRotation        = 60;

legendAll           = {};
for i = 1:nObservationNoises
    legendAll{i}    = ['\sigma_m = ', num2str(loopVars.observationNoises(i))];
end

%******************************************
% create the coverage and F ratio figure, which combines intercept and  slope metrics

%*** top row - intercept coupled
subplot(2,3,1);
create_aboxplot(metrics_intercept, 'int',       nObservationNoises, model_names, fontRotation, fontSize);
%ylim([-.1 1.1]); 
ylim([0 1.1]); 
title('Intercept coverage'); 

subplot(2,3,2);
create_aboxplot(metrics_intercept, 'slope',     nObservationNoises, model_names, fontRotation, fontSize);
%ylim([-.1 1.1]); 
ylim([0 1.1]); 
title('Slope coverage'); 

subplot(2,3,3);
create_aboxplot(metrics_intercept, 'fRatio',    nObservationNoises, model_names, fontRotation, fontSize);
%ylim([-5 1.1]); 
ylim([-3 1.1]);
title('F ratio'); 

 
%*** bottom row - slope coupled
subplot(2,3,4);
create_aboxplot(metrics_slope, 'int',       nObservationNoises, model_names, fontRotation, fontSize);
%ylim([-.1 1.1]); 
ylim([0 1.1]); 
title('Intercept coverage'); 

subplot(2,3,5);
create_aboxplot(metrics_slope, 'slope',     nObservationNoises, model_names, fontRotation, fontSize);
%ylim([-.1 1.1]); 
ylim([0 1.1]); 
title('Slope coverage'); 

subplot(2,3,6);
create_aboxplot(metrics_slope, 'fRatio',    nObservationNoises, model_names, fontRotation, fontSize);
%ylim([-5 1.1]);
ylim([-3 1.1]);
title('F ratio'); 
 
leg = legend(legendAll{:});
set(leg, 'Position', [0.8 0.8 0.1 0.1], 'Units', 'normalized'); %set(gca, 'FontSize', 13);
%leg.FontSize = 14;

%*****************************
function create_aboxplot(m, field, nObservationNoises, model_names, fontRotation, fontSize)

X           = {};
for j = 1:nObservationNoises
    X{j}    = m{1, j}.(field);
end
aboxplot(X, 'labels', model_names, 'colorgrad','green_down', 'plotMean', false); rotateFigFont(fontRotation); set(gca, 'FontSize', fontSize);

