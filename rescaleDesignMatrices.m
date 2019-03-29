function [dataTrain_out, dataTest_out]            = rescaleDesignMatrices(dataTrain, dataTest, normStats)
    
dataTrain_out               = dataTrain;
dataTest_out                = dataTest;

% %do this before rescaling the matrix
% X                             = blkdiag(dataTrain.designMat_cell{:});
% invA_ols                      = inv(X'*X);
% modelOutput.indep.twoStd_pos  = modelOutput.indep.m + 2 * sqrt(diag(invA_ols));
% modelOutput.indep.twoStd_neg  = modelOutput.indep.m - 2 * sqrt(diag(invA_ols));

[iTrain, iTest]            	= deal(1);
for i = 1:n_tasks
    train_i               	= dataTrain.designMat_cell{i};
    test_i                	= dataTest.designMat_cell{i}; 
    for j = 2:(P+1)
        train_i(:, j)     	= (train_i(:, j) * designMat_std(j)) + designMat_mean(j);
        test_i(:, j)       	=  (test_i(:, j) * designMat_std(j)) + designMat_mean(j);
    end
    dataTrain.designMat_cell{i} = train_i;
    dataTest.designMat_cell{i}	= test_i;
    
    iTrain                	= iTrain + nSamples_train(i);
    iTest                 	= iTest  + nSamples_test(i);
end
dataTrain_out.designMat_cell = dataTrain.designMat_cell;
dataTest_out.designMat_cell  = dataTest.designMat_cell;


%*** rescale coeffs
% in normalized variables: y = a * x1_norm + b * x2_norm + c  
% in original   variables: y = a/sigma1 * x1 + b/sigma2 * x2 + [c - a*mu1/sigma1 - b*mu2/sigma2]
% ... can be generalized to n variables trivially
%

% invA                                    = modelOutput.post.invA;
% if size(invA, 2) ~= 1
%     invA                                = diag(invA);
% end
% modelOutput.post.twoStd_pos             = modelOutput.post.m + 2 * sqrt(invA);
% modelOutput.post.twoStd_neg             = modelOutput.post.m - 2 * sqrt(invA);
% 

%NOTE: to keep it simple, just keeping the diagnoal part of parameter uncertainty
%modelOutput.post.invA                   = diag(modelOutput.post.invA);

for i = 2:(P+1)
    int_update_i                        = (modelOutput.post.m(i:(P+1):end) * designMat_mean(i))/designMat_std(i);
    %update intercepts
    modelOutput.post.m(1:(P+1):end)             = modelOutput.post.m(1:(P+1):end)           - int_update_i;  
    %modelOutput.post.twoStd_pos(1:(P+1):end)    = modelOutput.post.twoStd_pos(1:(P+1):end)  - int_update_i;
    %modelOutput.post.twoStd_neg(1:(P+1):end)    = modelOutput.post.twoStd_neg(1:(P+1):end)  - int_update_i;

    %update current coeffs
    modelOutput.post.m(i:(P+1):end)             = modelOutput.post.m(i:(P+1):end)         	/ designMat_std(i);
    %modelOutput.post.twoStd_pos(i:(P+1):end)  	= modelOutput.post.twoStd_pos(i:(P+1):end)	/ designMat_std(i);
    %modelOutput.post.twoStd_neg(i:(P+1):end)  	= modelOutput.post.twoStd_neg(i:(P+1):end)	/ designMat_std(i);

    modelOutput.post.invA(i:(P+1):end)          = modelOutput.post.invA(i:(P+1):end)/(designMat_std(i)^ 2); %alternative: *


    %****************** indep
    %update intercepts
    int_update_i_indep                       	= (modelOutput.indep.m(i:(P+1):end) * designMat_mean(i))/designMat_std(i);
    modelOutput.indep.m(1:(P+1):end)          	= modelOutput.indep.m(1:(P+1):end) - int_update_i_indep;
    modelOutput.indep.twoStd_pos(1:(P+1):end)   = modelOutput.indep.twoStd_pos(1:(P+1):end) - int_update_i_indep;
    modelOutput.indep.twoStd_neg(1:(P+1):end)   = modelOutput.indep.twoStd_neg(1:(P+1):end) - int_update_i_indep;

    %update current coeffs
    modelOutput.indep.m(i:(P+1):end)             = modelOutput.indep.m(i:(P+1):end)             / designMat_std(i);
    modelOutput.indep.twoStd_pos(i:(P+1):end)    = modelOutput.indep.twoStd_pos(i:(P+1):end)	/ designMat_std(i);
    modelOutput.indep.twoStd_neg(i:(P+1):end)  	 = modelOutput.indep.twoStd_neg(i:(P+1):end)	/ designMat_std(i);

end

