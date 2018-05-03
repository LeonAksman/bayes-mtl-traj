function [varargout] = blr_diag_test(hyp, X, t, nTasks, numBlocks, extraKernels, xs)

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

VERSION             = 'inverse'; %'inverse' or 'chol'

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

assert(D == nTasks * numBlocks);


NOISE_MIN           = 1e-6;  %1e-7;  

beta                = 1; %exp(hyp(1));

%******* added
% alpha_diag          = exp(hyp(1)) + NOISE_MIN;
% dispf('alpha_diag: %.1f', alpha_diag);
% dSigma              = kron(eye(nTasks), eye(numBlocks)); 
% dHyper              = 1; %exp(hyp(1)); 
% 
% Sigma             	= alpha_diag * kron(eye(nTasks), eye(numBlocks));
% %*********** EXTRA
% invSigma            = (1/alpha_diag) * kron(eye(nTasks), eye(numBlocks));
%******** CHANGE HERE ********
  
alpha               = exp(hyp(1));
Sigma               = 1./alpha*eye(D);  % weight prior covariance
invSigma            = alpha*eye(D);     % weight prior precision
dSigma              = -alpha^-2*eye(D);
dHyper              = alpha;
%*************************************************************************

dispf('1/alpha: %f', 1/alpha);

% useful quantities
XX                  = X'*X;
cholSigma        	= chol(Sigma); %** upper triagle


switch VERSION
    case 'chol'
        
        v        	= X * Sigma;
        chol_temp  	= chol((1/beta) * eye(N) + v*X');

        
        invA      	= Sigma - v' * solve_chol(chol_temp, v);
        %invA    	= Sigma - v' * ((1/beta) * eye(N) + v*X')\ v;           %potentially better?
        
    case 'inverse'
        A       	= beta*XX + inv(Sigma);           % posterior precision 
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

    invSigma_m      = solve_chol(cholSigma, m);
    
%     nlZ_const       = D*log(2*pi); %N*log(2*pi);   
%     nlZ             = -0.5*( N*log(beta) - nlZ_const - logdetSigma ...
%                       - beta*(t-X*m)'*(t-X*m) - m'* invSigma_m - ...
%                       logdetA );
    %REMOVE NEGATIVE?
    nlZ = -0.5*( N*log(beta) - N*log(2*pi) - logdetSigma ...
                 - beta*(t-X*m)'*(t-X*m) - m'*invSigma*m - ...
                 logdetA );
             
    if nargout > 1    % derivatives?
        dnlZ        = 0;
        b           = (eye(D) - beta*Q*X)*Q*t;
        
        % repeatedly computed quantities for derivatives
        Xt          = X'*t;
        XXm         = XX*m;
        
        %SXt         = S*Xt;
        SXt         = invA * Xt;
                         
                        
           
        invSigma_dSigma  	=  solve_chol(cholSigma, dSigma);
        F                   = -solve_chol(cholSigma, invSigma_dSigma')';

        c                   = -beta*S*F*SXt;
       
%         dnlZ                = -( -0.5*trace(invSigma_dSigma) + ...
%                                 beta*Xt'*c - beta*c'*XXm - c'*invSigma_m - ...
%                                 0.5*m'*F*m - 0.5*trace(invA * F) ) * dHyper;            
% dnlZ = 1;

        %**** andre's, REMOVE NEGATIVE?
        dnlZ = -( -0.5*sum(sum(invSigma.*dSigma')) + ...
               beta*Xt'*c - beta*c'*XXm - c'*invSigma*m - ...
               0.5*m'*F*m - 0.5*trace(A\F) )*dHyper;

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