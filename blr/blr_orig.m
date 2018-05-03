function [varargout] = blr_orig(hyp, X, t, xs)

% Bayesian linear regression
%
% Fits a bayesian linear regression model, where the inputs are:
%    hyp : vector of hyperparmaters. hyp = [log(beta); log(alpha)]
%    X   : N x D data matrix
%    t   : N x 1 vector of targets 
%    xs  : Nte x D matrix of test cases
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

if nargin<3 || nargin>4
    disp('Usage: [nlZ dnlZ] = blr(hyp, x, y);')
    disp('   or: [mu  s2  ] = blr(hyp, x, y, xs);')
    return
end

[N,D]  = size(X);
beta   = exp(hyp(1));     % noise precision
alpha  = exp(hyp(2:end)); % weight precisions
Nalpha = length(alpha);
if Nalpha ~= 1 && Nalpha ~= D
    error('hyperparameter vector has invalid length');
end

if Nalpha == D
    Sigma  = spdiags(1./alpha,0,D,D);
    invSigma = spdiags(alpha,0,D,D);
else    
    Sigma  = 1./alpha*speye(D);  % weight prior covariance
    invSigma = alpha*speye(D);     % weight prior precision
end

% useful quantities
XX = X'*X;
A  = beta*XX + invSigma;     % posterior precision
S  = inv(A);                 % posterior covariance
Q  = S*X';
m  = beta*Q*t;               % posterior mean

% compute like this for to avoid numerical overflow
logdetA     = 2*sum(log(diag(chol(A))));
logdetSigma = sum(log(diag(A)));            % assumes Sigma is diagonal

if nargin == 3
    nlZ = -0.5*( N*log(beta) - N*log(2*pi) - logdetSigma ...
                 - beta*(t-X*m)'*(t-X*m) - m'*invSigma*m - ...
                 logdetA );
    
    if nargout > 1    % derivatives?
        dnlZ = zeros(size(hyp));
        b    = (eye(D) - beta*Q*X)*Q*t;
        
        % repeatedly computed quantities for derivatives
        Xt  = X'*t;
        XXm = XX*m;
        SXt = S*Xt;
        
        % noise precision
        dnlZ(1) = -( N/(2*beta) - 0.5*(t'*t) + t'*X*m + beta*t'*X*b - 0.5*m'*XXm ...
                     - beta*b'*XXm - b'*invSigma*m -0.5*trace(Q*X) )*beta;
         
        % variance parameters
        for i = 1:Nalpha
            if Nalpha == D % use ARD?
                dSigma    = sparse(i,i,-alpha(i)^-2,D,D);
                dinvSigma = sparse(i,i,1,D,D);
            else
                dSigma    = -alpha(i)^-2*speye(D);
                dinvSigma = speye(D);
            end
            
            F = dinvSigma;
            c = -beta*S*F*SXt;
            
            dnlZ(i+1) = -( -0.5*sum(sum(invSigma.*dSigma')) + ...
                           beta*Xt'*c - beta*c'*XXm - c'*invSigma*m - ...
                           0.5*m'*F*m - 0.5*trace(A\F) )*alpha(i);
        end
        post.m = m;
        post.A = A;
    end
    if nargout > 1
        varargout = {nlZ, dnlZ, post};
    else
        varargout = {nlZ};
    end
    
else % prediction mode
    ys     = xs*m;
    s2     = 1/beta + diag(xs*(A\xs'));
    post.m = m;
    post.A = A;
    varargout = {ys, s2, post};
end

end