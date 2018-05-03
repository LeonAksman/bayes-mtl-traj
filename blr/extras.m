% A                   = beta*XX + inv(Sigma);
% m2                	= beta*inv(A)*X'*t;
% % 
% invA1               = Sigma - v' * inv((1/beta) * eye(N) + v*X') * v;
% m3               	= beta * invA1 * X' * t;
% 
% 
% invAx                = Sigma - v' * solve_chol(chol_temp, v);
% close all;
% figure(1); imagesc(A * inv(A)); 
% figure(2); imagesc(A * invA); 
% figure(3); imagesc(A * invAx);