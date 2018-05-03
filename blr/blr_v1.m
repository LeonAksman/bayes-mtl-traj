function [varargout] = blr_v1(hyp, X, t, nTasks, nDims, xs, nTasks_test)

% Bayesian linear regression
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
%    X   : N x (nTasks * nDims)             data matrix
%    t   : N x 1                            vector of targets across all tasks
%    xs  : Ntest x (nTasks_test * nDims)    matrix of test cases
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

if nargin<5 || nargin>7
    disp('Usage: [nlZ dnlZ] = blr(hyp, x, y, nTasks, nDims);')
    disp('   or: [mu  s2  ] = blr(hyp, x, y, nTasks, nDims, xs, nTasks_test);')
    return
end

%[N,D]  = size(X);
[N, nTasks_D]       = size(X);

assert(nTasks_D == nTasks * nDims);

assert(length(hyp) == 3);

beta                = exp(hyp(1));           % noise precision
alpha               = exp(hyp(2));         	% weight precision
gamma               = invLogit(hyp(3));      % coupling trade-off parameter

% Nalpha = length(alpha);
% if Nalpha ~= 1 && Nalpha ~= D
%     error('hyperparameter vector has invalid length');
% end
%
% if Nalpha == D
%     Sigma  = spdiags(1./alpha,0,D,D);
%     invSigma = spdiags(alpha,0,D,D);
% else    
%     Sigma  = 1./alpha*speye(D);  % weight prior covariance
%     invSigma = alpha*speye(D);     % weight prior precision
% end

Sigma               = kron((1/alpha) * (gamma * eye(nTasks) + (1 - gamma) * ones(nTasks)), ones(nDims));
invSigma            = inv(Sigma);

dSigma_dAlpha       = -(1/alpha) * Sigma;
dInvSigma_dAlpha    = (1/alpha) * invSigma;
dHyper_alpha        = alpha;

dSigma_dGamma       = (1/alpha) * kron(eye(nTasks) - ones(nTasks), ones(nDims));
dInvSigma_dGamma    = - invSigma * dSigma_dGamma * invSigma;
dHyper_gamma        = gamma * (1 - gamma);

dSigmas             = {dSigma_dAlpha,       dSigma_dGamma};
dInvSigmas          = {dInvSigma_dAlpha,    dInvSigma_dGamma};
dHypers             = {dHyper_alpha,        dHyper_gamma};

%********************************

% useful quantities
XX                  = X'*X;
A                   = beta*XX + invSigma;     % posterior precision
S                   = inv(A);                 % posterior covariance
Q                   = S*X';
m                   = beta*Q*t;               % posterior mean

% compute like this for to avoid numerical overflow
logdetA             = 2*sum(log(diag(chol(A))));

%logdetSigma         = sum(log(diag(A)));            % assumes Sigma is diagonal
logdetSigma         = 2*sum(log(diag(chol(Sigma))));    

if nargin == 5
    nlZ             = -0.5*( N*log(beta) - N*log(2*pi) - logdetSigma ...
                      - beta*(t-X*m)'*(t-X*m) - m'*invSigma*m - ...
                      logdetA );
    
    if nargout > 1    % derivatives?
        dnlZ        = zeros(size(hyp));
        b           = (eye(D) - beta*Q*X)*Q*t;
        
        % repeatedly computed quantities for derivatives
        Xt          = X'*t;
        XXm         = XX*m;
        SXt         = S*Xt;
        
        % noise precision
        dnlZ(1)     = -( N/(2*beta) - 0.5*(t'*t) + t'*X*m + beta*t'*X*b - 0.5*m'*XXm ...
                      - beta*b'*XXm - b'*invSigma*m -0.5*trace(Q*X) )*beta;
                 
         
%         % variance parameters
%         for i = 1:Nalpha
%             
%             if Nalpha == D % use ARD?
%                 dSigma    = sparse(i,i,-alpha(i)^-2,D,D);
%                 dinvSigma = sparse(i,i,1,D,D);
%             else
%                 dSigma    = -alpha(i)^-2*speye(D);
%                 dinvSigma = speye(D);
%             end
%             
%             F = dinvSigma;
%             c = -beta*S*F*SXt;
%             
%             dnlZ(i+1) = -( -0.5*sum(sum(invSigma.*dSigma')) + ...
%                            beta*Xt'*c - beta*c'*XXm - c'*invSigma*m - ...
%                            0.5*m'*F*m - 0.5*trace(A\F) )*alpha(i);
%         end
        
        
        % variance parameters
        for i = 1:length(dSigmas)
            
            dSigma 	= dSigmas{i};
            dHyper 	= dHypers{i};
                        
            F     	= dInvSigmas{i};
            c    	= -beta*S*F*SXt;
            
            dnlZ(i+1) = -( -0.5*sum(sum(invSigma.*dSigma')) + ...
                           beta*Xt'*c - beta*c'*XXm - c'*invSigma*m - ...
                           0.5*m'*F*m - 0.5*trace(A\F) ) * dHyper;
        end
        
        post.m      = m;
        post.A      = A;
    end
    if nargout > 1
        varargout   = {nlZ, dnlZ, post};
    else
        varargout   = {nlZ};
    end
    
else % prediction mode
    
    %assert();
    
    ys              = xs*m;
    s2              = 1/beta + diag(xs*(A\xs'));
    post.m          = m;
    post.A          = A;
    varargout       = {ys, s2, post};
end

end