function [varargout] = blr_mtl_mkl_test2(hyp, X, t, nTasks, nDimsPerTask, extraKernels, xs)

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
% Fits a bayesian linear regression model, where the sinputs are:
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

n_extra_kernels     = length(extraKernels);
%assert(length(hyp) == (3 + n_extra_kernels));

[N, D]              = size(X);

assert(D == nTasks * nDimsPerTask);


beta                = exp(hyp(1));           % noise precision

alpha1              = max(1000*eps, exp(hyp(2)));
alpha2              = exp(hyp(3));
couplingMat         = alpha1 * eye(nTasks) + (alpha2/nTasks) * ones(nTasks);
for i = 1:n_extra_kernels
    alphaExtra_i    = exp(hyp(3 + i));
    couplingMat     = couplingMat + alphaExtra_i * extraKernels(i).mat;
end
index_alphaDims     = (3 + n_extra_kernels + 1):(3 + n_extra_kernels + nDimsPerTask);
alphaDims           = exp(hyp(index_alphaDims));

%********* CHANGED
D_model           	= diag(alphaDims);  %eye(nDimsPerTask)

Sigma               = kron(couplingMat, D_model);


cholSigma        	= chol(Sigma); %** upper triagle


%*********** alpha1
dSigma_dAlpha1    	= kron(eye(nTasks), D_model);
dHyper_alpha1      	= alpha1;

%*********** alpha2
dSigma_dAlpha2     	= kron((1/nTasks) * ones(nTasks), D_model);
dHyper_alpha2      	= alpha2;

dSigmas             = {dSigma_dAlpha1,       dSigma_dAlpha2};
dHypers             = {dHyper_alpha1,        dHyper_alpha2};

%********** extra kernels
for i = 1:n_extra_kernels
    dSigma_dAlpha_i  = kron(extraKernels(i).mat, D_model);
    
    alphaExtra_i    = exp(hyp(3 + i));
    dHyper_alpha_i   = alphaExtra_i;
    
    dSigmas{end+1} 	= dSigma_dAlpha_i;
    dHypers{end+1}  = dHyper_alpha_i;
end

%********* diagonal hypers
for i = 1:length(alphaDims)
    
    v_i             = zeros(length(alphaDims), 1);
    v_i(i)          = 1;
    
    dSigmas{end+1} 	= kron(couplingMat, v_i * v_i');
    dHypers{end+1}  = alphaDims(i);    
end

%********************************

% useful quantities
XX                  = X'*X;

%A                   = beta*XX + invSigma;           % posterior precision
v                   = X * Sigma;
chol_temp           = chol((1/beta) * eye(N) + v*X');
invA                = Sigma - v' * solve_chol(chol_temp, v);

S                   = invA;
Q                   = S*X';
m                   = beta*Q*t;               % posterior mean

% compute like this for to avoid numerical overflow
logdetSigma         = 2*sum(log(diag(cholSigma)));   
logdetA             = -logdetSigma + N*log(beta) + 2*sum(log(diag(chol_temp))); %2*sum(log(diag(cholA)));         
 

if nargin == 6

    invSigma_m      = solve_chol(cholSigma, m);
    
    nlZ_const       = D*log(2*pi); %N*log(2*pi);
    
    nlZ             = -0.5*( N*log(beta) - nlZ_const - logdetSigma ...
                      - beta*(t-X*m)'*(t-X*m) - m'* invSigma_m - ...
                      logdetA );

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
            
            dSigma 	= dSigmas{i};
            dHyper 	= dHypers{i};
                        
           
            invSigma_dSigma  	=  solve_chol(cholSigma, dSigma);
            F                   = -solve_chol(cholSigma, invSigma_dSigma')';
            
            c    	= -beta*S*F*SXt;
            
            dnlZ(i+1) = -( -0.5*trace(invSigma_dSigma) + ...
                           beta*Xt'*c - beta*c'*XXm - c'*invSigma_m - ...
                           0.5*m'*F*m - 0.5*trace(invA * F) ) * dHyper;            
            
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