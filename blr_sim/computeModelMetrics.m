function [metrics, model_names]  = computeModelMetrics(inParams, f_generatePredictionStructure, groupDifferencesStyle)
%inParams: n_tasks, rngSeed

[predictStruct, traj_coeffs, true_diffs]    = feval(f_generatePredictionStructure, inParams);


%************* training/testing indices cells
[uniqueSubjects, index_first]     	= unique(predictStruct.subj_id, 'first');
nUniqueSubjects                     = length(uniqueSubjects);
[indicesCell_train, ...
 indicesCell_test]                  = deal(cell(nUniqueSubjects, 1));
[nTrain_vec, age_bl]              	= deal(zeros(nUniqueSubjects, 1));
for i = 1:nUniqueSubjects 
    index_i                         = find(predictStruct.subj_id == uniqueSubjects(i));
    n_i                             = length(index_i);
  
    %must be some data for training
  	assert(n_i  > inParams.numTestSamples);
  
    indicesCell_train{i}          	= index_i(1:(n_i  - inParams.numTestSamples));
	indicesCell_test{i}          	= index_i((n_i  - inParams.numTestSamples + 1):n_i);
    
    nTrain_vec(i)                   = length(indicesCell_train{i});
    age_bl(i)                       = predictStruct.age(index_i(1));
end
%***********************************************

intercepts                      = traj_coeffs(:, 1);
slopes                          = traj_coeffs(:, 2);

couplingInfo_noiseLess          = asRowCol(true_diffs, 'col'); 
couplingInfo_noisy              = couplingInfo_noiseLess + normrnd(0, inParams.biomarkerNoise, size(couplingInfo_noiseLess));

similarityKernel                = squareform(pdist(couplingInfo_noisy, 'squaredeuclidean'));
similarityKernel                = similarityKernel/norm(similarityKernel);

kernelSubjectIds             	= predictStruct.unique_subj_id;

% [~, index_bl]                   = unique(predictStruct.subj_id, 'first');
% age_bl_vec                      = vertcat(indicesCell_train{:});

%************** create the models
t_vec                           = age_bl;               %repVec(age_bl,                nTrain_vec);
biomarker_vec                   = couplingInfo_noisy;   %repVec(couplingInfo_noisy,    nTrain_vec);
models                          = formModels(kernelSubjectIds, t_vec, biomarker_vec);

%***** compute and plot
for i = 1:length(models)
    dispf('**** running model: %s', models(i).name);
    
    %models(i).out          	= predict_blr_mtl_flex(predictStruct, models(i));
    
 	[models(i).modelOutput,   ...
     models(i).predictStruct, ...
     models(i).evalOutput]      = runAnalysis(predictStruct, models(i), indicesCell_train, indicesCell_test); %runAnalysis_mtl
    
    dispf('**********************');
    
end

n_m                         = length(models);
model_names                 = {};

NUM_STDS                    = 2;
    
 
for i = 1:(n_m+1)
    
    if i <= length(models)
        
     	%**** prediction coverage
        target_i           	= models(i).evalOutput.sim.targets;
        pred_i            	= models(i).evalOutput.sim.preds;
        predStds_i        	= sqrt(models(i).evalOutput.sim.predVars);
        assert(all(predStds_i > 0));

        is_good_pred_i    	= target_i >= (pred_i - NUM_STDS*predStds_i) & ...
                              target_i <= (pred_i + NUM_STDS*predStds_i);
        metrics.predCoverage(i)	= sum(is_good_pred_i)  /length(is_good_pred_i);

    	%**** parameter coverage 
        int_i            	= models(i).modelOutput.sim.post.m(1:2:end);
        slope_i           	= models(i).modelOutput.sim.post.m(2:2:end);
        
        [invA_dim1, invA_dim2] = size(models(i).modelOutput.sim.post.invA);
        if invA_dim1 == invA_dim2
            all_stds      	= sqrt(diag(models(i).modelOutput.sim.post.invA));
        else
            all_stds      	= sqrt(models(i).modelOutput.sim.post.invA);
        end
        
        mae_i               = models(i).evalOutput.sim.mae;
        logML_i             = NaN;
        if isfield(models(i).modelOutput.sim, 'logML')
            logML_i         = models(i).modelOutput.sim.logML;
        end
        
        modelName_i         = models(i).name;
        model_names{end+1}  = models(i).name_short;
        
    	%*** ints within 2 stds
        int_i_std         	= all_stds(1:2:end);
        is_good_int_i    	= intercepts  >= (int_i - NUM_STDS*int_i_std) & ...
                              intercepts  <= (int_i + NUM_STDS*int_i_std);

        %*** slopes within 2 stds
        slope_i_std          	= all_stds(2:2:end);
        is_good_slope_i       	= slopes  >= (slope_i - NUM_STDS*slope_i_std) & ...
                                  slopes  <= (slope_i + NUM_STDS*slope_i_std);

% %     	%*** ints within 2 stds
%         int_i_2std_pos     	= models(i).modelOutput.sim.post.twoStd_pos(1:2:end);
%         int_i_2std_neg     	= models(i).modelOutput.sim.post.twoStd_neg(1:2:end);
%         is_good_int_i2    	= intercepts  >= int_i_2std_neg & intercepts  <= int_i_2std_pos;
% 
%         %*** slopes within 2 stds
%         slope_i_2std_pos   	= models(i).modelOutput.sim.post.twoStd_pos(2:2:end);
%         slope_i_2std_neg   	= models(i).modelOutput.sim.post.twoStd_neg(2:2:end);
%         is_good_slope_i2   	= slopes  >= slope_i_2std_neg & slopes  <= slope_i_2std_pos;
%         
%         assert(all(is_good_slope_i == is_good_slope_i2) & all(is_good_int_i == is_good_int_i2));
        
    else %**** OLS part
        
        %**** prediction coverage
        target_i           	= models(1).evalOutput.sim.targets;
        pred_i            	= models(1).evalOutput.sim.preds_ols;
        predStds_i        	= sqrt(models(1).evalOutput.sim.predVars_ols);

        assert(all(predStds_i > 0));

        is_good_pred_i    	= target_i >= (pred_i - NUM_STDS*predStds_i) & ...
                              target_i <= (pred_i + NUM_STDS*predStds_i);
        metrics.predCoverage(i)	= sum(is_good_pred_i)  /length(is_good_pred_i);
        
        int_i            	= models(1).modelOutput.sim.indep.m(1:2:end);
        slope_i           	= models(1).modelOutput.sim.indep.m(2:2:end);

    	%*** ints within 2 stds
        int_i_2std_pos     	= models(1).modelOutput.sim.indep.twoStd_pos(1:2:end);
        int_i_2std_neg     	= models(1).modelOutput.sim.indep.twoStd_neg(1:2:end);
        is_good_int_i    	= intercepts  >= int_i_2std_neg & intercepts  <= int_i_2std_pos;

        %*** slopes within 2 stds
        slope_i_2std_pos   	= models(1).modelOutput.sim.indep.twoStd_pos(2:2:end);
        slope_i_2std_neg   	= models(1).modelOutput.sim.indep.twoStd_neg(2:2:end);
        is_good_slope_i   	= slopes  >= slope_i_2std_neg & slopes  <= slope_i_2std_pos;
        
        mae_i               = models(1).evalOutput.sim.mae_ols;
        logML_i             = NaN; 
        
        modelName_i         = 'OLS';
      	model_names{end+1}  = 'OLS';
    end

    metrics.intCoverage(i)   	= sum(is_good_int_i)  /length(is_good_int_i);
    metrics.slopeCoverage(i) 	= sum(is_good_slope_i)/length(is_good_slope_i);
       
    metrics.mae(i)             = log10(mae_i);
    metrics.intMae(i)          = log10(mean(abs(int_i   - intercepts)));
    metrics.slopeMae(i)        = log10(mean(abs(slope_i - slopes))); 
    
    [p_slope_i, t_slope_i]	= anova1(slope_i,   true_diffs', 'off');
    [p_int_i,   t_int_i]   	= anova1(int_i,     true_diffs', 'off');
    F_slope_i               = t_slope_i{2, 5};
    F_int_i                 = t_int_i{2, 5};
    
    F_slope_i               = conditional(isnan(F_slope_i), 0, F_slope_i);
    F_int_i                 = conditional(isnan(F_int_i),   0, F_int_i);
    
    switch groupDifferencesStyle
        case 'intercept'
            metrics.fRatio(i)  = (F_int_i - F_slope_i) / F_int_i;
            corrVar         = intercepts;
        case 'slope'
            metrics.fRatio(i)  = (F_slope_i - F_int_i) / F_slope_i;
            corrVar         = slopes;
    end
    
    dispf('%5d%40s: COVERAGE: int %5.2f / slope: %5.2f,  CORR: int %5.2f, slope %5.2f, logML: %5.1f, F int, %5.1f, F slope %5.1f, F Ratio: %10.1f', ...
        i,                          ...
        modelName_i,                ...
        metrics.intCoverage(i),     ...
        metrics.slopeCoverage(i),   ...
        corr(corrVar, int_i),      	...
        corr(corrVar, slope_i),   	...
        logML_i,                    ...
        F_int_i,                    ...
        F_slope_i,                  ...        
        metrics.fRatio(i));
end
