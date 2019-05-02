function [varargout] = blr_mtl_mkl_inv(hyp, X, t, params, extraKernels, xs)

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
%    params       : structure containing:
%       nTasks    : number of tasks (e.g. subjects)
%       nBlocks   : the number of dimensions in each task's model, so that D = nTasks * numBlocks
%       noiseMin  : minimum value along the diagonal of the parameter covariance matrix (default 1e-6)
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
if nargin<5 || nargin>6
    disp('Usage: [nlZ dnlZ] = blr_mtl_mkl_flex(hyp, X, t, params, extraKernels);')
    disp('   or: [mu  s2  ] = blr_mtl_mkl_flex(hyp, X, t, params, extraKernels, xs);')
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

% %initialize hyperparameter vector to correct size
% if (length(hyp) == 1 && all(hyp == 0)) || isempty(hyp)
%     nHyp          	= 2 + (2 + nKernels + numInternalHypers) * numBlocks;     	%beta + alpha_diag + (alpha1, alpha2 + kernel weights) x number of blocks
%     hyp           	= zeros(nHyp, 1);
% end

[N, D]              = size(X);
[dSigmas, dHypers] 	= deal({});
blocknum            = [];

nTasks              = params.nTasks;
nBlocks             = params.nBlocks;
if isfield(params, 'noiseMin')
    NOISE_MIN     	= params.noiseMin;
else
    NOISE_MIN     	= 1e-6;
end

assert(D == nTasks * nBlocks);

beta                = exp(hyp(1));

%******* added
alpha_diag          = exp(hyp(2)) + NOISE_MIN; 

dSigmas{end+1}      = kron(eye(nTasks), eye(nBlocks)); 
dHypers{end+1}      = exp(hyp(2)); 
blocknum(end+1)     = 0;

currPos             = 3; %2
  

numHypsPerBlock     = 2 + nKernels + numInternalHypers;

Sigma               = alpha_diag * kron(eye(nTasks), eye(nBlocks));
Sigma_blocks       	= cell(nBlocks, 1);

for i = 1:nBlocks
    
    Sigma_blocks{i}    = alpha_diag * eye(nTasks);
    
    index_i         = currPos:(currPos + numHypsPerBlock - 1);
    currPos         = currPos + numHypsPerBlock;
    
    exp_hyps_i    	= exp(hyp(index_i));
  
    alpha1          = exp_hyps_i(1);
    alpha2          = exp_hyps_i(2);
     
 	v_i          	= zeros(nBlocks, 1);
    v_i(i)        	= 1;
    delta_i       	= v_i * v_i';
    
    %*** indep
    dSigmas{end+1} 	= kron(eye(nTasks),  delta_i);
    dHypers{end+1} 	= alpha1;   
    blocknum        = [blocknum; i];
    
    %*** fully coupled
    dSigmas{end+1} 	= kron(ones(nTasks), delta_i);    
    dHypers{end+1}	= alpha2;
    blocknum        = [blocknum; i];
    
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
                blocknum        = [blocknum; i];

            
            case 'gaussian'
                alpha_kernel_j  = exp_hyps_i(currHyp);      
                currHyp         = currHyp + 1;
                
                sigma_j         = exp_hyps_i(currHyp);       
                currHyp         = currHyp + 1;

                kernel_j        = exp(-sigma_j * mat_j);    
 
                %*** external hyper
                dSigmas{end+1} 	= kron(kernel_j,  delta_i);
                dHypers{end+1} 	= alpha_kernel_j; 
                blocknum        = [blocknum; i];

            
                %*** internal hyper
                r_squared       = mat_j;
                dSigmas{end+1} 	= kron(-alpha_kernel_j * r_squared .* kernel_j, delta_i);
                dHypers{end+1}  = sigma_j;
                blocknum        = [blocknum; i];

        end 
        
        assert(~any(isnan(kernel_j(:))), sprintf('kernel %d has NaNs in it', j));
        coupling_i              = coupling_i + alpha_kernel_j * kernel_j;

    end    
    
    Sigma_blocks{i}           	= Sigma_blocks{i} + coupling_i;
    Sigma                       = Sigma + kron(coupling_i, delta_i);
end

%*************************************************************************

% useful quantities
XX                  = X'*X;

%cholSigma           = chol(Sigma); %** upper triagle
cholBlocks          = cellfun(@chol, Sigma_blocks, 'UniformOutput', false);
cholSigma           = blkdiag(cholBlocks{:});

switch VERSION
    case 'chol'  
        tic;
        v        	= X * Sigma;
        chol_temp  	= chol((1/beta) * eye(N) + v*X');

        
        invA      	= Sigma - v' * solve_chol(chol_temp, v);
        toc;
    case 'inverse'

        %invSigma    = inv(Sigma);
        invBlocks    = cellfun(@inv, Sigma_blocks, 'UniformOutput', false);
        invSigma     = blkdiag(invBlocks{:});

        
        A       	= beta*XX + invSigma;           % posterior precision 
        invA        = inv(A);
end

S                   = invA;
%Q                   = S*X';

%*** faster
%m                   = beta*Q*t;                  	% posterior mean
m                   = beta*S*(X'*t);              	% posterior mean

% compute like this for to avoid numerical overflow
logdetSigma         = 2*sum(log(diag(cholSigma)));   

switch VERSION
    case 'chol'
        logdetA  	= -logdetSigma + N*log(beta) + 2*sum(log(diag(chol_temp))); 
    case 'inverse'
        cholA       = chol(A);
        logdetA  	= 2*sum(log(diag(cholA)));  
end

if nargin == 5

    switch VERSION
        case 'chol' 
            invSigma_m      = solve_chol(cholSigma, m);
        case 'inverse'
            invSigma_m      = invSigma * m;
    end
    
    nlZ_const       = N*log(2*pi); 
    
    nlZ             = -0.5*( N*log(beta) - nlZ_const - logdetSigma ...
                      - beta*(t-X*m)'*(t-X*m) - m'* invSigma_m - ...
                      logdetA );

    if nargout > 1    % derivatives?
        dnlZ        = zeros(size(hyp));
        
     	% repeatedly computed quantities for derivatives
        Xt          = X'*t;
        XXm         = XX*m;
        SXt         = invA * Xt;
         
        %**** much faster
        %b           = (eye(D) - beta*Q*X)*Q*t;
        b        	= SXt - beta*S*(XX*SXt);      %same as: Qt - beta*Q*(X*Qt); 
        
        % noise precision
        dnlZ(1)     = -( N/(2*beta) - 0.5*(t'*t) + t'*X*m + beta*t'*X*b - 0.5*m'*XXm ...
                      - beta*b'*XXm - b'*invSigma_m -0.5*trace(S*XX) )*beta;  %trace(S*XX) == trace(Q*X)
                  
        % variance parameters
        for i = 1:length(dSigmas)
            
            dSigma_i 	= dSigmas{i};
            dHyper_i 	= dHypers{i};
                        

            switch VERSION
                case 'chol'
                    invSigma_dSigma  	=  solve_chol(cholSigma, dSigma_i);
                    F                   = -solve_chol(cholSigma, invSigma_dSigma')';
                case 'inverse'
                    %invSigma_dSigma  	=  invSigma * dSigma_i;
                    %F                   = -invSigma * invSigma_dSigma'; 
                    
                    if blocknum(i) == 0
                        invSigma_dSigma  	= invSigma;
                        F_cell              = cellfun(@(x1, x2) -x1*x2, invBlocks, invBlocks, 'UniformOutput', false);
                        F                   = blkdiag(F_cell{:});
                    else
                        j                                   = blocknum(i);
                        block_i                             = ((j-1)*nTasks + 1):(j*nTasks);
                        invSigma_dSigma                 	= zeros(size(invSigma));
                        invSigma_dSigma(block_i, block_i)	= invBlocks{j}*dSigma_i(block_i, block_i);
                        F                               	= zeros(size(invSigma));
                        F(block_i, block_i)                 = -invBlocks{j}*invSigma_dSigma(block_i, block_i)';
                    end
            end           

            %**** much faster
            %c    	= -beta*S*F*SXt;
            c      = -beta*S*(F*SXt);
            
            %*** break the trace up into blocks because F is block diagonal
            %tr_invA_F = trace(invA * F);
            tr_invA_F = 0;
            for j = 1:nBlocks
                
                %F is fully block diagonal for first covariance hyperparam,
                %but only has a single non-zero block for all the rest
                if blocknum(i) == 0 || blocknum(i) == j
                    index_j    = ((j-1)*nTasks + 1):(j*nTasks);
                    tr_invA_F  = tr_invA_F + trace(invA(index_j,index_j) * F(index_j,index_j));
                end
            end 
            
            dnlZ(i+1) = -( -0.5*trace(invSigma_dSigma) + ...
                           beta*Xt'*c - beta*c'*XXm - c'*invSigma_m - ...
                           0.5*m'*F*m - 0.5*tr_invA_F ) * dHyper_i;         
        end   
       
        post.beta 	= beta;
        post.alpha  = exp(hyp(2:end));
        post.m      = m;
        post.invA   = invA;
        post.invSigma   = invSigma;        
    end
    if nargout > 1
        varargout   = {nlZ, dnlZ, post};
    else
        varargout   = {nlZ};
    end
    
else % prediction mode
      
    %re-indexing the prediction design matrix
    xs_orig          	= xs;
    xs               	= [];
    for i = 1:nBlocks
        xs            	= [xs xs_orig(:, i:nBlocks:end)];
    end
    
    %get the predictive mean using reindex design mat
    ys              = xs*m;
    
    %get the predictive uncertainty using reindexed design mat
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
        s2                      = 1/beta + diag(xs*(invA * xs'));
    end
    
    post.beta       = beta;
    post.alpha      = exp(hyp(2:end));
    post.m          = m;
    post.invA       = invA;
    post.invSigma   = invSigma;
    varargout       = {ys, s2, post};
end

end