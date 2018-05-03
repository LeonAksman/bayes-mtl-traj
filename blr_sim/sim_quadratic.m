function sim_quadratic

addpath '../utils';
addpath '../blr';


loopVars.tasks                       	= 100;  %200 
loopVars.observationNoises           	= 1;    %[0.1 1 2 4 8];
loopVars.biomarkerNoises            	= 1;
loopVars.numRngs                    	= 1;

saveFilename                            = '~/Documents/MATLAB/sim_interceptCoupling_200_varyObsNoise.mat';

loopParams(saveFilename, loopVars, @generatePredictionStructure_intercepts, 'intercept');
