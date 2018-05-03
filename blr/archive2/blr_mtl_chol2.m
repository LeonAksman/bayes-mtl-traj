function [varargout] = blr_mtl_chol2(hyp, X, t, nTasks, nDimsPerTask, xs)

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
% Updated by L. Aksman for new parameterization of prior covariance that enables multi-task learning

if nargin<5 || nargin>6
    disp('Usage: [nlZ dnlZ] = blr(hyp, x, y, nTasks, nDims);')
    disp('   or: [mu  s2  ] = blr(hyp, x, y, nTasks, nDims, xs);')
    return
end

[N, D]              = size(X);

assert(D == nTasks * nDimsPerTask);

%assert(length(hyp) == 3);

beta                = exp(hyp(1));           % noise precision
alpha               = exp(hyp(2));         	 % weight precision
gamma               = invLogit(hyp(3));      % coupling trade-off parameter

%****** CHANGED
%Sigma              = kron((1/alpha) * ((1 - gamma) * ones(nTasks) +          gamma * eye(nTasks)), eye(nDimsPerTask));
Sigma               = kron((1/alpha) * ((1 - gamma) * ones(nTasks) + nTasks * gamma * eye(nTasks)), eye(nDimsPerTask));
cholSigma        	= chol(Sigma); %** upper triagle

%invSigma            = L_Sigma' \ (L_Sigma \ eye(size(Sigma)));       %********************* inv(Sigma);       


%*********** alpha
dSigma_dAlpha       = -(1/alpha) * Sigma;
%dInvSigma_dAlpha    = (1/alpha) * invSigma;
dHyper_alpha        = alpha;

%*********** gamma
%dSigma_dGamma      = (1/alpha) * kron(-ones(nTasks) +          eye(nTasks), eye(nDimsPerTask));
dSigma_dGamma       = (1/alpha) * kron(-ones(nTasks) + nTasks * eye(nTasks), eye(nDimsPerTask));
%dInvSigma_dGamma    = - invSigma * dSigma_dGamma * invSigma;
dHyper_gamma        = gamma * (1 - gamma);

dSigmas             = {dSigma_dAlpha,       dSigma_dGamma};
%dInvSigmas          = {dInvSigma_dAlpha,    dInvSigma_dGamma};
dHypers             = {dHyper_alpha,        dHyper_gamma};

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
 

if nargin == 5
%     nlZ             = -0.5*( N*log(beta) - N*log(2*pi) - logdetSigma ...
%                       - beta*(t-X*m)'*(t-X*m) - m'*invSigma*m - ...
%                       logdetA );
    invSigma_m      = solve_chol(cholSigma, m);
    
    nlZ             = -0.5*( N*log(beta) - N*log(2*pi) - logdetSigma ...
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
        %dnlZ(1)     = -( N/(2*beta) - 0.5*(t'*t) + t'*X*m + beta*t'*X*b - 0.5*m'*XXm ...
        %              - beta*b'*XXm - b'*invSigma*m -0.5*trace(Q*X) )*beta;
        dnlZ(1)     = -( N/(2*beta) - 0.5*(t'*t) + t'*X*m + beta*t'*X*b - 0.5*m'*XXm ...
                      - beta*b'*XXm - b'*invSigma_m -0.5*trace(Q*X) )*beta;
                  
        % variance parameters
        for i = 1:length(dSigmas)
            
            dSigma 	= dSigmas{i};
            dHyper 	= dHypers{i};
                        
            %F     	=  dInvSigmas{i};
            
            invSigma_dSigma  	=  solve_chol(cholSigma, dSigma);
            F                   = -solve_chol(cholSigma, invSigma_dSigma')';
            
            c    	= -beta*S*F*SXt;
            
            %dnlZ(i+1) = -( -0.5*sum(sum(invSigma.*dSigma')) + ...
            %               beta*Xt'*c - beta*c'*XXm - c'*invSigma*m - ...
            %               0.5*m'*F*m - 0.5*trace(A\F) ) * dHyper;
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
    %s2              = 1/beta + diag(xs*(A\xs'));
    s2              = 1/beta + diag(xs*(invA * xs'));
    post.m          = m;
    post.invA       = invA;
    varargout       = {ys, s2, post};
end

end