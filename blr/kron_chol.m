function [M, cholA, cholB] = kron_chol(A, B)
%Finds chol(kron(A, B))
%
%from: http://www.math.uwaterloo.ca/~hwolkowi/henry/reports/kronthesisschaecke04.pdf


% cholA   = chol(A);
% cholB   = chol(B);
% 
% kron_AB = kron(cholA', cholB);
% 
% M       = kron_AB' * kron_AB;

cholA = chol(A);
cholB = chol(B);
M     = kron(cholA, cholB);