function [varargout] = blr_mtl_mkl_inv_split(hyp, X, t, nTasks, numBlocks, extraKernels, xs)

% A Bayesian linear regression based approach to multi-task learning with multi-kernel based coupling.
% Computations use matrix inverses in this version.
%
% ***************************************
% Assumed covariance prior structure: 
%
%   alpha1 * eye + sum_i ( kron(sigma_ci, M_ii) )
%
% where: sigma_ci = alpha_i1 * eye(n) + alpha_i2 * ones(n) + sum_j ( alpha_i(j+2) K(j) )
%        M_ii has one in (i, i)th element, zero otherwise
% ***************************************
%
% Fits a bayesian linear regression model, where the inputs are:
%    hyp          : vector of hyperparmaters. hyp = [log(beta); log(alpha); logit(gamma)]
%    X            : N     x D                 data matrix
%    t            : N     x 1                 vector of targets across all tasks
%    nTasks       : number of tasks (e.g. subjects)
%    numBlocks    : the number of dimensions in each task's model, so that D = nTasks * numBlocks
%    extraKernels : a structure for the coupling kernels K in the prior
%    xs           : Ntest x (nTasks * nDims)  matrix of test cases
% 
%  where N = sum(N_i), N_i is number of targets per task
%
%
% Two modes are supported: 
%    [nlZ, dnlZ, post] = blr_mtl_mkl_inv(hyp, x, t, ...);      % report evidence and derivatives
%    [mu, s2, post]    = blr_mtl_mkl_inv(hyp, x, t, ..., xs);  % predictive mean and variance
%
% Written by L.Aksman based on code provided by A. Marquand

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
[dSigmas, dHypers] 	= deal({});

assert(D == nTasks * numBlocks);


NOISE_MIN           = 1e-6;
NOISE_MIN_OUTLIER   = 1e-3; 

%************* ASSUMPTION: all hyperparams are > 0, so we optimize log(hyp), which can vary between -Inf to Inf
exp_hyps            = exp(hyp);

beta                = exp_hyps(1);

%******* added
alpha_diag          = exp_hyps(2) + NOISE_MIN; 
dSigmas{end+1}      = kron(eye(nTasks), eye(numBlocks)); 
dHypers{end+1}      = exp_hyps(2); 


currPos             = 2; 
  

%numHypsPerBlock     = 2 + nKernels + numInternalHypers;


Sigma               = alpha_diag * kron(eye(nTasks), eye(numBlocks)); %zeros(D, D);

for i = 1:numBlocks
    
    currPos         = currPos + 1;
    alpha1          = exp_hyps(currPos);    %conditional(~isempty(extraKernels) && ~any(ismember(i, extraKernels(1).blocks)), 0, exp_hyps(currPos)); %
    
    currPos         = currPos + 1;
    alpha2          = exp_hyps(currPos);    %conditional(~isempty(extraKernels) && ~any(ismember(i, extraKernels(1).blocks)), 0, exp_hyps(currPos));

    
 	v_i          	= zeros(numBlocks, 1);
    v_i(i)        	= 1;
    delta_i       	= v_i * v_i';
    
    %*** indep
    dSigmas{end+1} 	= kron(eye(nTasks),  delta_i);
    dHypers{end+1} 	= alpha1;   
    %*** fully coupled
    dSigmas{end+1} 	= kron(ones(nTasks), delta_i);  
    dHypers{end+1}	= alpha2;
    
    coupling_i     	= alpha1 * eye(nTasks) + (alpha2/nTasks) * ones(nTasks); 
    for j = 1:nKernels

        if ~any(ismember(i, extraKernels(j).blocks))
            currPos             = currPos + numHyp_kernel(extraKernels);
            
        	dSigmas{end+1}      = [];
            dHypers{end+1}      = []; 
            continue;
        end
        
        mat_j                   = extraKernels(j).mat;
        type_j                  = extraKernels(j).type;

        switch type_j
            case 'linear'
             	kernel_j        = mat_j;
                
             	currPos         = currPos + 1;
                alpha_kernel_j  = exp_hyps(currPos);
                %alpha_kernel_j  = exp_hyps_i(currHyp);
                %currHyp         = currHyp + 1;

                dSigmas{end+1} 	= kron(kernel_j,  delta_i);
                dHypers{end+1} 	= alpha_kernel_j; 
                
                coupling_i    	= coupling_i + alpha_kernel_j * kernel_j;
                
            case 'gaussian'
  
                currPos         = currPos + 1;
                alpha_kernel_j  = exp_hyps(currPos);
                %alpha_kernel_j  = exp_hyps_i(currHyp);      
                %currHyp         = currHyp + 1;
                
                currPos         = currPos + 1;
                sigma_j         = exp_hyps(currPos);     
                %sigma_j         = exp_hyps_i(currHyp);       
                %currHyp         = currHyp + 1;

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
    
    nlZ_const       = N*log(2*pi); %D*log(2*pi); %
    
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
                        
            if isempty(dSigma_i) && isempty(dHyper_i)
                dnlZ(i+1) = 0;
                continue;
            end
            
            switch VERSION
                case 'chol'
                    invSigma_dSigma  	=  solve_chol(cholSigma, dSigma_i);
                    F                   = -solve_chol(cholSigma, invSigma_dSigma')';
                case 'inverse'
                    invSigma_dSigma  	=  invSigma * dSigma_i;
                    F                   = -invSigma * invSigma_dSigma'; %-solve_chol(cholSigma, invSigma_dSigma')';
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
    
    numSamples      = size(xs, 1);
    if numSamples > 1000
        %do it in chunks
        chunkSize   = 1000;
        numChunks   = round(numSamples/chunkSize);
        
        s2          = zeros(numSamples, 1);
        for i = 1:numChunks
            if i < numChunks
                index_i = ((i-1)*chunkSize + 1):(i*chunkSize);
            else
                index_i = ((i-1)*chunkSize + 1):numSamples;
            end
            s2(index_i) = 1/beta + diag(xs(index_i, :)*(invA * xs(index_i, :)'));
        end
        
    else
        s2        	= 1/beta + diag(xs*(invA * xs'));
    end
    post.m          = m;
    post.invA       = invA;
    varargout       = {ys, s2, post};
end

end
