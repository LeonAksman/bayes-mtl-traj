function models                 = formModels(kernelSubjectIds, t, noisyBiomarker)

if nargin < 3
    t                           = [];
end

%***** random
rng(1);
randvec                         = rand(length(noisyBiomarker), 1);
random_gauss                    = squareform(pdist(randvec, 'squaredeuclidean'));
random_gauss                    = random_gauss / norm(random_gauss);


similarityKernel                = squareform(pdist(noisyBiomarker, 'squaredeuclidean'));
similarityKernel                = similarityKernel/norm(similarityKernel);

%*************** default params
MIN_SAMPLES_TRAINING            = 2;

% commonParams.P                  = 1;
% commonParams.mode               = 'predict_last';
% commonParams.minTrainingSamples = MIN_SAMPLES_TRAINING;
% commonParams.extraKernels       = [];
% commonParams.kernelSubjectIds   = kernelSubjectIds;
% commonParams.f_blr              = @blr_mtl_mkl_inv_reindex;         %@blr_mtl_mkl_inv_split;
% commonParams.f_optimizer      	= @minimize;
% commonParams.maxeval            = -100; %-200; %
% commonParams.standardize        = true;

commonParams.extraKernels       = [];
commonParams.kernelSubjectIds   = kernelSubjectIds;
commonParams.normDesignMatrix  	= true;
commonParams.normTargets        = true;
commonParams.f_train            = @train_mtl;
commonParams.f_predict        	= @predict_mtl;
commonParams.f_eval             = @eval_mtl;
commonParams.f_optimizer      	= @minimize_gpml;
commonParams.f_blr              = @blr_mtl_mkl_inv_reindex;
commonParams.maxeval            = -50; %-100; 
commonParams.P                  = 1;

%**************** specify models
models                          = [];

clear model;
model.name                      = 'Coupled, random';
model.name_short                = 'random';
model.params                    = commonParams;
model.params.extraKernels(1).mat        = random_gauss;
model.params.extraKernels(1).blocks    	= [1 2];   
model.params.extraKernels(1).type    	= 'gaussian';   
model.params.extraKernels(1).bound    	= 'positive'; 
models                          = [models; model];

clear model;
model.name                      = 'Coupled, linear noisy (both)';
model.name_short                = 'linear both';
model.params                    = commonParams;
model.params.extraKernels(1).mat        = noisyBiomarker * noisyBiomarker';
model.params.extraKernels(1).blocks    	= [1 2];   
model.params.extraKernels(1).type    	= 'linear';   
model.params.extraKernels(1).bound    	= 'positive'; 
models                          = [models; model];

clear model;
model.name                      = 'Coupled, gaussian noisy (both)';
model.name_short                = 'Gaussian both';
model.params                    = commonParams;
model.params.extraKernels(1).mat        = similarityKernel;
model.params.extraKernels(1).blocks    	= [1 2];   
model.params.extraKernels(1).type    	= 'gaussian';   
model.params.extraKernels(1).bound    	= 'positive'; 
models                          = [models; model];

clear model;
model.name                      = 'Coupled, linear noisy (intercept)';
model.name_short                = 'linear int';
model.params                    = commonParams;
model.params.f_blr              = @blr_mtl_mkl_inv_split;
model.params.extraKernels(1).mat        = noisyBiomarker * noisyBiomarker';
model.params.extraKernels(1).blocks    	= [1];   
model.params.extraKernels(1).type    	= 'linear';   
model.params.extraKernels(1).bound    	= 'positive'; 
models                          = [models; model];

clear model;
model.name                      = 'Coupled, gaussian noisy (intercept)';
model.name_short                = 'Gaussian int';
model.params                    = commonParams;
model.params.f_blr              = @blr_mtl_mkl_inv_split;
model.params.extraKernels(1).mat        = similarityKernel;
model.params.extraKernels(1).blocks    	= [1];   
model.params.extraKernels(1).type    	= 'gaussian';   
model.params.extraKernels(1).bound    	= 'positive'; 
models                          = [models; model];

clear model;
model.name                      = 'Coupled, linear noisy (slope)';
model.name_short                = 'linear slope';
model.params                    = commonParams;
model.params.f_blr              = @blr_mtl_mkl_inv_split;
model.params.extraKernels(1).mat        = noisyBiomarker * noisyBiomarker';
model.params.extraKernels(1).blocks    	= [2];   
model.params.extraKernels(1).type    	= 'linear';   
model.params.extraKernels(1).bound    	= 'positive'; 
models                          = [models; model];

clear model;
model.name                      = 'Coupled, gaussian noisy (slope)';
model.name_short                = 'Gaussian slope';
model.params                    = commonParams;
model.params.f_blr              = @blr_mtl_mkl_inv_split;
model.params.extraKernels(1).mat        = similarityKernel;
model.params.extraKernels(1).blocks    	= [2];   
model.params.extraKernels(1).type    	= 'gaussian';   
model.params.extraKernels(1).bound    	= 'positive'; 
models                          = [models; model];

clear model;
model.name                      = 'Coupled';
model.name_short                = 'plain';
model.params                    = commonParams;
models                          = [models; model];

%************************************************************
%
% LME
%
%************************************************************
% t                               = zscore(t);
% noisyBiomarker                  = zscore(noisyBiomarker);

clear model;
model.name               	= 'rI';
model.name_short          	= 'LME: rI';
model.params              	= commonParams;
model.params.normDesignMatrix = false;
model.params.normTargets      = false;
model.params.fixedEffects     = [t noisyBiomarker]; %[age_true_bl amyloid_first]; 
model.params.randomEffectsVec = [1];                   %random int
model.params.f_train       	= @train_fsLME;
model.params.f_predict   	= @predict_fsLME;
model.params.f_eval       	= @eval_fsLME;
models                      = [models; model];

clear model;
model.name               	= 'rI_rS';
model.name_short          	= 'LME: rI,rS';
model.params              	= commonParams;
model.params.normDesignMatrix = false;
model.params.normTargets      = false;
model.params.fixedEffects     = [t noisyBiomarker]; %[age_true_bl amyloid_first]; 
model.params.randomEffectsVec = [1 2];            	%random int, random slope
model.params.f_train       	= @train_fsLME;
model.params.f_predict   	= @predict_fsLME;
model.params.f_eval       	= @eval_fsLME;
models                      = [models; model];

% clear model;
% model.name               	= 'rI_rS_rBio';
% model.name_short          	= 'LME: rI,rS,rBio';
% model.params              	= commonParams;
% model.params.fixedEffects     = [t noisyBiomarker]; %[age_true_bl amyloid_first]; 
% model.params.randomEffectsVec = [1 2 3];            	%random int, random slope, random biomarker
% model.params.f_train       	= @train_fsLME;
% model.params.f_predict   	= @predict_fsLME;
% model.params.f_eval       	= @eval_fsLME;
% models                      = [models; model];


