function createMAEFig(figureNum, loopVars, metrics_intercept, metrics_slope, model_names_int, model_names_slope)

addpath '../aboxplot';

addpath '../subtightplot';
subplot                         = @(m,n,p) subtightplot (m, n, p, [0.17 0.05], [0.15 0.05], [0.04 0.02]);


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
% top row - intercept coupled
figure(figureNum);

for i = 1:4
    subplot(2,4,i);
    create_aboxplot(metrics_intercept{1, i, 1}.mae, model_names_int, fontRotation, fontSize);
    title(['log_1_0MAEs, \sigma_m = ' num2str(loopVars.observationNoises(i))]);
end

%*** bottom row - slope coupled
for i = 1:4
    subplot(2,4,i+4);
    create_aboxplot(metrics_slope{1, i, 1}.mae,     model_names_slope, fontRotation, fontSize);
  	title(['log_1_0MAEs, \sigma_m = ' num2str(loopVars.observationNoises(i))]);
end

%tightfigadv;
%saveFigureToPng(1, '~/Desktop/papers/mtl_blr/fig_simulations_MAE.png');