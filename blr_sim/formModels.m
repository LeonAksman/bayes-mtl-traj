function models                 = formModels(kernelSubjectIds, noisyBiomarker)

%***** random
rng(1);
randvec                         = rand(length(noisyBiomarker), 1);
random_gauss                    = squareform(pdist(randvec, 'squaredeuclidean'));
random_gauss                    = random_gauss / norm(random_gauss);


similarityKernel                = squareform(pdist(noisyBiomarker, 'squaredeuclidean'));
similarityKernel                = similarityKernel/norm(similarityKernel);

%*************** default params
MIN_SAMPLES_TRAINING            = 2;
commonParams.P                  = 1;
commonParams.mode               = 'predict_last';
commonParams.minTrainingSamples = MIN_SAMPLES_TRAINING;
commonParams.extraKernels       = [];
commonParams.kernelSubjectIds   = kernelSubjectIds;
commonParams.f_blr              = @blr_mtl_mkl_inv_split;   %@blr_mtl_mkl_inv;
commonParams.f_optimizer      	= @minimize;
commonParams.maxeval            = -100; %-200; %
commonParams.standardize        = true;

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
model.params.extraKernels(1).mat        = noisyBiomarker * noisyBiomarker';
model.params.extraKernels(1).blocks    	= [1];   
model.params.extraKernels(1).type    	= 'linear';   
model.params.extraKernels(1).bound    	= 'positive'; 
models                          = [models; model];

clear model;
model.name                      = 'Coupled, gaussian noisy (intercept)';
model.name_short                = 'Gaussian int';
model.params                    = commonParams;
model.params.extraKernels(1).mat        = similarityKernel;
model.params.extraKernels(1).blocks    	= [1];   
model.params.extraKernels(1).type    	= 'gaussian';   
model.params.extraKernels(1).bound    	= 'positive'; 
models                          = [models; model];

clear model;
model.name                      = 'Coupled, linear noisy (slope)';
model.name_short                = 'linear slope';
model.params                    = commonParams;
model.params.extraKernels(1).mat        = noisyBiomarker * noisyBiomarker';
model.params.extraKernels(1).blocks    	= [2];   
model.params.extraKernels(1).type    	= 'linear';   
model.params.extraKernels(1).bound    	= 'positive'; 
models                          = [models; model];

clear model;
model.name                      = 'Coupled, gaussian noisy (slope)';
model.name_short                = 'Gaussian slope';
model.params                    = commonParams;
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
