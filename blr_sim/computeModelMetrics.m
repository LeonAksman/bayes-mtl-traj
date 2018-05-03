function [metrics, model_names]  = computeModelMetrics(inParams, f_generatePredictionStructure, groupDifferencesStyle)
%inParams: n_tasks, rngSeed

[predictStruct, traj_coeffs, true_diffs]    = feval(f_generatePredictionStructure, inParams);

intercepts                      = traj_coeffs(:, 1);
slopes                          = traj_coeffs(:, 2);

couplingInfo_noiseLess          = asRowCol(true_diffs, 'col'); 
couplingInfo_noisy              = couplingInfo_noiseLess + normrnd(0, inParams.biomarkerNoise, size(couplingInfo_noiseLess));


%[p_theory, t_theory]            = anova1(couplingInfo_noisy, true_diffs, 'off');
%F_theoretical                   = t_theory{2, 5};

similarityKernel                = squareform(pdist(couplingInfo_noisy, 'squaredeuclidean'));
similarityKernel                = similarityKernel/norm(similarityKernel);

kernelSubjectIds             	= predictStruct.unique_subj_id;
models                          = formModels(kernelSubjectIds, couplingInfo_noisy);

%***** compute and plot
for i = 1:length(models)
    dispf('**** running model: %s', models(i).name);
    models(i).out          	= predict_blr_mtl_flex(predictStruct, models(i));
    dispf('**********************');
    
end

n_m                         = length(models);
model_names                 = {};

NUM_STDS                    = 2;
    
 
for i = 1:(n_m+1)
    if i <= length(models)
        int_i            	= models(i).out.sim.post.m(1:2:end);
        slope_i           	= models(i).out.sim.post.m(2:2:end);
        
        [invA_dim1, invA_dim2] = size(models(i).out.sim.post.invA);
        if invA_dim1 == invA_dim2
            all_stds      	= sqrt(diag(models(i).out.sim.post.invA));
        else
            all_stds      	= sqrt(models(i).out.sim.post.invA);
        end
        
        mae_i               = models(i).out.sim.mae_mtl;
        logML_i             = models(i).out.sim.logML;
        
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


        
    else %**** OLS part
        int_i            	= models(1).out.sim.indep.m(1:2:end);
        slope_i           	= models(1).out.sim.indep.m(2:2:end);
        
%         X                   = blkdiag(models(1).out.sim.designMat_cell{:});
%         invA_ols            = inv(X'*X);
%         all_stds           	= sqrt(diag(invA_ols));
%         
%         if models(1).out.sim.normalizeDesignMatrix
%             P               = models(1).params.P;
%          	for j = 2:(P+1)
%                  %not squared.. this is std. dev.
%                 all_stds(j:(P+1):end)	= all_stds(j:(P+1):end) / (models(1).out.sim.designMat_std(j)); %alternative: *
%             end
%         end

    	%*** ints within 2 stds
        int_i_2std_pos     	= models(1).out.sim.indep.twoStd_pos(1:2:end);
        int_i_2std_neg     	= models(1).out.sim.indep.twoStd_neg(1:2:end);
        is_good_int_i    	= intercepts  >= int_i_2std_neg & intercepts  <= int_i_2std_pos;

        %*** slopes within 2 stds
        slope_i_2std_pos   	= models(1).out.sim.indep.twoStd_pos(2:2:end);
        slope_i_2std_neg   	= models(1).out.sim.indep.twoStd_neg(2:2:end);
        is_good_slope_i   	= slopes  >= slope_i_2std_neg & slopes  <= slope_i_2std_pos;
        
        mae_i               = models(1).out.sim.mae_indep;
        logML_i             = NaN; %models(i).out.sim.logML;
        
        modelName_i         = 'OLS';
      	model_names{end+1}  = 'OLS';
    end

    metrics.int(i)         	= sum(is_good_int_i)  /length(is_good_int_i);
    metrics.slope(i)      	= sum(is_good_slope_i)/length(is_good_slope_i);
    metrics.mae(i)          = mae_i;
    
    [p_slope_i, t_slope_i]	= anova1(slope_i,   true_diffs', 'off');
    [p_int_i,   t_int_i]   	= anova1(int_i,     true_diffs', 'off');
    F_slope_i               = t_slope_i{2, 5};
    F_int_i                 = t_int_i{2, 5};
    
    switch groupDifferencesStyle
        case 'intercept'
            metrics.fRatio(i)  = (F_int_i - F_slope_i) / F_int_i; %F_int_i/F_slope_i;
            corrVar         = intercepts;
        case 'slope'
            metrics.fRatio(i)  = (F_slope_i - F_int_i) / F_slope_i;
            corrVar         = slopes;
    end
    
    dispf('%5d%40s: COVERAGE: int %5.2f / slope: %5.2f,  CORR: int %5.2f, slope %5.2f, logML: %5.1f, F int, %5.1f, F slope %5.1f, F Ratio: %10.1f', ...
        i,                          ...
        modelName_i,                ...
        metrics.int(i),             ...
        metrics.slope(i),           ...
        corr(corrVar, int_i),      	...
        corr(corrVar, slope_i),   	...
        logML_i,                    ...
        F_int_i,                    ...
        F_slope_i,                  ...        
        metrics.fRatio(i));
end

%metrics.models                  = models;
%metrics.model_names             = model_names;

%SCALE_MAE                   = 1;
%plot_models(models, SCALE_MAE);