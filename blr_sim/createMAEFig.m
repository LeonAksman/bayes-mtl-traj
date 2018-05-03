function createMAEFig(figureNum, loopVars, metrics_intercept, metrics_slope, model_names)

addpath '../aboxplot';

%***********************************************
% setup stuff

nTasks               = size(loopVars.tasks, 1);
nObservationNoises   = size(loopVars.observationNoises, 2);
nBiomarkerNoises     = size(loopVars.biomarkerNoises, 3);

assert(nBiomarkerNoises == 1);

%close all; 

fontSize            = 25; %15
fontRotation        = 60;

legendAll           = {};
for i = 1:nObservationNoises
    legendAll{i}    = ['\sigma_m = ', num2str(loopVars.observationNoises(i))];
end

%******************************************
% top row - intercept coupled

%*** add a '*' to indicate oracle models
model_names_int     = model_names;
model_names_int{4} 	= [model_names{4} '*'];
model_names_int{5} 	= [model_names{5} '*'];

figure(figureNum);

for i = 1:4
    subplot(2,4,i);
    create_aboxplot(metrics_intercept{1, i, 1}.mae, model_names_int, fontRotation, fontSize);
    title(['MAEs, \sigma_m = ' num2str(loopVars.observationNoises(i))]);
end

%*** bottom row - slope coupled

%*** add a '*' to indicate oracle models
model_names_slope    = model_names;
model_names_slope{6} 	= [model_names{6} '*'];
model_names_slope{7} 	= [model_names{7} '*'];

for i = 1:4
    subplot(2,4,i+4);
    create_aboxplot(metrics_slope{1, i, 1}.mae,     model_names_slope, fontRotation, fontSize);
  	title(['MAEs, \sigma_m = ' num2str(loopVars.observationNoises(i))]);
end

%tightfigadv;
%saveFigureToPng(1, '~/Desktop/papers/mtl_blr/fig_simulations_MAE.png');