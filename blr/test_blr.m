function test_blr()




%generate samples from noisy polynomial function

n               = 50;
t_final         = 10;
noise_true     	= 20;

Z               = [ones(n,1) linspace(0, t_final, n)'];
coeffs_true       = [1; 1];

X               = Z * coeffs_true + noise_true * rand(n, 1);


%**************************************

log_beta      	= 0; 
log_alpha    	= 0;
hyp          	= [log_beta; log_alpha];

% function [varargout] = blr(hyp, X, t, xs)
% 
% Bayesian linear regression
%
% Fits a bayesian linear regression model, where the inputs are:
%    hyp : vector of hyperparmaters. hyp = [log(beta); log(alpha)]
%    X   : N x D data matrix
%    t   : N x 1 vector of targets 
%    xs  : Nte x D matrix of test cases

%Train
maxeval                     = -20;
hyp_pre                     = hyp;
[hyp,nlmls]                 = minimize_quiet(hyp, @blr, maxeval, Z, X);

%Test
[mu, s2, post]              = blr(hyp, Z, X, Z); % predictive mean and variance

close all;

plot(X);
hold on;
plot(mu,            '-g');
plot(mu + sqrt(s2), '-r');
plot(mu - sqrt(s2), '-r');

%**********************************************************

%     t_i                        	= X_all(:, i);
% 
%     log_beta                    = 0; 
%     log_alpha                   = 0;
%     logit_gamma                 = 0; %1000;
%     hyp                         = [log_beta; log_alpha; logit_gamma];
% 
%     nTasks                      = n;
%     nDims                       = P + 1;
%     
%     %Train
%     maxeval                     = -20; %-5
%     hyp_pre                     = hyp;
%     [hyp,nlmls]                 = minimize_quiet(hyp, @blr_mtl, maxeval, Z_all, t_i, nTasks, nDims);
% 
%     %Test
%     [mu, s2, post]              = blr_mtl(hyp, Z_all, t_i, nTasks, nDims, Z_all); % predictive mean and variance
% 
