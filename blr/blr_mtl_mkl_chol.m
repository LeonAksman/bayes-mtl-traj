function [varargout] = blr_mtl_mkl_chol(hyp, X, t, nTasks, numBlocks, extraKernels, xs)

% Bayesian linear regression: multi-task learning (MTL) version
%
% This version uses the Cholesky Decomposition based approach to inverting
% matrices
%
% ***************************************
% Assumed covariance prior structure: 
%
%    1/alpha * kron(gamma * eye(nTasks) + (1 - gamma) * ones(nTasks)), eye(nDims))  
%
% where: alpha > 0,  0 < gamma < 1
% ***************************************
%
% Fits a bayesian linear regression model, where the inputs are:
%    hyp : vector of hyperparmaters. hyp = [log(beta); log(alpha); logit(gamma)]
%    X   : N     x (nTasks * nDims)  data matrix
%    t   : N     x 1                 vector of targets across all tasks
%    xs  : Ntest x (nTasks * nDims)  matrix of test cases
% 
%  where N = sum(N_i), N_i is number of targets per task
%
% The hyperparameter beta is the noise precision and alpha is the precision
% over lengthscale parameters. This can be either a scalar variable (a
% common lengthscale for all input variables), or a vector of length D (a
% different lengthscale for each input variable, derived using an automatic
% relevance determination formulation).
%
% Two modes are supported: 
%    [nlZ, dnlZ, post] = blr(hyp, x, y);  % report evidence and derivatives
%    [mu, s2, post]    = blr(hyp, x, y, xs); % predictive mean and variance
%
% Written by A. Marquand
% Updated by L. Aksman for new parameterization of prior covariance that enables multi-task + multi-kernel learning

if nargin<6 || nargin>7
    disp('Usage: [nlZ dnlZ] = blr_mtl_mkl_flex(hyp, X, t, nTasks, nDimsPerTask, extraKernels);')
    disp('   or: [mu  s2  ] = blr_mtl_mkl_flex(hyp, X, t, nTasks, nDimsPerTask, extraKernels, xs);')
    return
end

VERSION             = 'chol'; %'inverse' or 'chol'

%check kernel properties
nKernels            = length(extraKernels);
if nKernels > 0
    kernelTypes         = {extraKernels.type};
    kernelBounds        = {extraKernels.bound};
    assert(all(strcmp(kernelBounds, 'positive')));
    
    numInternalHypers   = sum(strcmp(kernelTypes, 'gaussian'));
else
    numInternalHypers   = 0;
end

[N, D]              = size(X);
[dSigmas, dHypers] 	= deal({});

assert(D == nTasks * numBlocks);


NOISE_MIN           = 1e-6;
NOISE_MIN_OUTLIER   = 1e-3; 

beta                = exp(hyp(1));

%******* added
alpha_diag          = exp(hyp(2)) + NOISE_MIN; 
dSigmas{end+1}      = kron(eye(nTasks), eye(numBlocks)); 
dHypers{end+1}      = exp(hyp(2)); 


currPos             = 3; %2
  

numHypsPerBlock     = 2 + nKernels + numInternalHypers;


Sigma               = zeros(D, D);

for i = 1:numBlocks
    
    index_i         = currPos:(currPos + numHypsPerBlock - 1);
    currPos         = currPos + numHypsPerBlock;
    
    exp_hyps_i    	= exp(hyp(index_i));
  
    alpha1          = exp_hyps_i(1);
    alpha2          = exp_hyps_i(2);
     
 	v_i          	= zeros(numBlocks, 1);
    v_i(i)        	= 1;
    delta_i       	= v_i * v_i';
    
    dSigmas{end+1} 	= kron(eye(nTasks),  delta_i);
    dSigmas{end+1} 	= kron(ones(nTasks), delta_i); 
    dHypers{end+1} 	= alpha1;    
    dHypers{end+1}	= alpha2;
    
    currHyp         = 3;
    coupling_i     	= alpha1 * eye(nTasks) + (alpha2/nTasks) * ones(nTasks); 
    for j = 1:nKernels

        mat_j                   = extraKernels(j).mat;
        type_j                  = extraKernels(j).type;

        switch type_j
            case 'linear'
             	kernel_j        = mat_j;
                alpha_kernel_j  = exp_hyps_i(currHyp);
                currHyp         = currHyp + 1;

                dSigmas{end+1} 	= kron(kernel_j,  delta_i);
                dHypers{end+1} 	= alpha_kernel_j; 
                
                coupling_i    	= coupling_i + alpha_kernel_j * kernel_j;
                
            case 'gaussian'
                alpha_kernel_j  = exp_hyps_i(currHyp);      
                currHyp         = currHyp + 1;
                
                sigma_j         = exp_hyps_i(currHyp);       
                currHyp         = currHyp + 1;

                kernel_j        = exp(-sigma_j * mat_j);    
                
                %*** external hyper
                dSigmas{end+1} 	= kron(kernel_j,  delta_i);
                dHypers{end+1} 	= alpha_kernel_j; 
            
                %*** internal hyper
                r_squared       = mat_j;
                dSigmas{end+1} 	= kron(-alpha_kernel_j * r_squared .* kernel_j, delta_i);
                dHypers{end+1}  = sigma_j;
                                
                coupling_i    	= coupling_i + alpha_kernel_j * kernel_j;
        end       
    end    
    
    Sigma                       = Sigma + kron(coupling_i, delta_i);
end

Sigma                           = Sigma + alpha_diag * kron(eye(nTasks), eye(numBlocks));

%add in extra diagnoal term for outliers
if nKernels > 0 && any(strcmp({extraKernels.type}, 'outlier'))
    index_outlier             	= find(strcmp({extraKernels.type}, 'outlier'));
    if ~isempty(index_outlier)
        Sigma                 	= Sigma + NOISE_MIN_OUTLIER * kron(extraKernels(index_outlier).mat, eye(numBlocks));
    end
end
%*************************************************************************

% useful quantities
XX                  = X'*X;

%***************************** MOVED DOWN
cholSigma           = chol(Sigma); %** upper triagle

switch VERSION
    case 'chol'    
        v        	= X * Sigma;
        chol_temp  	= chol((1/beta) * eye(N) + v*X');

        
        invA      	= Sigma - v' * solve_chol(chol_temp, v);
        %invA    	= Sigma - v' * ((1/beta) * eye(N) + v*X')\ v;           %potentially better?
        
    case 'inverse'
        invSigma    = inv(Sigma);
        
        A       	= beta*XX + invSigma;           % posterior precision 
        invA        = inv(A);
end

S                   = invA;
Q                   = S*X';

m                   = beta*Q*t;               % posterior mean


% compute like this for to avoid numerical overflow
logdetSigma         = 2*sum(log(diag(cholSigma)));   

switch VERSION
    case 'chol'
        logdetA  	= -logdetSigma + N*log(beta) + 2*sum(log(diag(chol_temp))); 
    case 'inverse'
        cholA       = chol(A);
        logdetA  	= 2*sum(log(diag(cholA)));  
end

if nargin == 6

    switch VERSION
        case 'chol'
            invSigma_m      = solve_chol(cholSigma, m);
        case 'inverse'
            invSigma_m      = invSigma * m;
    end
    
    nlZ_const       = D*log(2*pi); %N*log(2*pi);
    
    nlZ             = -0.5*( N*log(beta) - nlZ_const - logdetSigma ...
                      - beta*(t-X*m)'*(t-X*m) - m'* invSigma_m - ...
                      logdetA ); % * conditional(bNEGATE, -1, 1);

    if nargout > 1    % derivatives?
        dnlZ        = zeros(size(hyp));
        b           = (eye(D) - beta*Q*X)*Q*t;
        
        % repeatedly computed quantities for derivatives
        Xt          = X'*t;
        XXm         = XX*m;
        
        %SXt         = S*Xt;
        SXt         = invA * Xt;
        
        % noise precision
        dnlZ(1)     = -( N/(2*beta) - 0.5*(t'*t) + t'*X*m + beta*t'*X*b - 0.5*m'*XXm ...
                      - beta*b'*XXm - b'*invSigma_m -0.5*trace(Q*X) )*beta; 
                  
        % variance parameters
        for i = 1:length(dSigmas)
            
            dSigma_i 	= dSigmas{i};
            dHyper_i 	= dHypers{i};
                        

            switch VERSION
                case 'chol'
                    invSigma_dSigma  	=  solve_chol(cholSigma, dSigma_i);
                    F                   = -solve_chol(cholSigma, invSigma_dSigma')';
                case 'inverse'
                    invSigma_dSigma  	=  invSigma * dSigma_i;
                    F                   = -invSigma * invSigma_dSigma'; %solve_chol(cholSigma, invSigma_dSigma')';
            end           
            %invSigma_dSigma  	=  solve_chol(cholSigma, dSigma_i);
            %F                   = -solve_chol(cholSigma, invSigma_dSigma')';
            
            c    	= -beta*S*F*SXt;
            
            dnlZ(i+1) = -( -0.5*trace(invSigma_dSigma) + ...
                           beta*Xt'*c - beta*c'*XXm - c'*invSigma_m - ...
                           0.5*m'*F*m - 0.5*trace(invA * F) ) * dHyper_i;         
            
        end   
        
        post.m      = m;
        post.invA   = invA;
    end
    if nargout > 1
        varargout   = {nlZ, dnlZ, post};
    else
        varargout   = {nlZ};
    end
    
else % prediction mode
      
    ys              = xs*m;
    s2              = 1/beta + diag(xs*(invA * xs'));
    post.m          = m;
    post.invA       = invA;
    varargout       = {ys, s2, post};
end

end