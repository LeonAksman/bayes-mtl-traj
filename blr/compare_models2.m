%*******************************************************************************
function compare_models2(models, diag_first)

n_models                  	= length(models);

fields                      = fieldnames(models(1).out);

for i = 1:length(fields)
    
    field_i                     = fields{i};
    
    %targets_i                 	= vertcat(models(1).out.(field_i).targetsTest_cell{:});    
    %est_indep_i              	= vertcat(models(1).out.(field_i).estimatesCell_indep{:});
    %X_indep                     = abs(est_indep_i - targets_i)';
    X_indep                     = abs(models(1).out.(field_i).err_indep)';
    
    X                           = [];
    for j = 1:n_models
        
        if isempty(models(j).out)
            break;
        end
        
        %est_j                   = vertcat(models(j).out.(field_i).estimatesCell_mtl{:});
        %X_j                     = abs(est_j - targets_i)';
        X_j                     = abs(models(j).out.(field_i).err_mtl)';
        
        X                       = [X; X_j];
    end
    X                       = [X; X_indep];

    n_rows                  = size(X, 1);
    
    M                       = eye(n_rows, n_rows);
    for k1 = 1:n_rows
        for k2 = 1:(k1-1)
            [~, ~, ~, stats]    	= ttest(X(k1, :), X(k2, :));
            M(k1,k2)                = stats.tstat;
 
            [~, ~, ~, stats]    	= ttest(X(k2, :), X(k1, :));
            M(k2,k1)                = stats.tstat;
            
            %[M(k1,k2), M(k2,k1)]  = deal(signrank(X(k1, :), X(k2, :)));
            %[M(k1,k2), M(k2,k1)]  = deal(anova1(X(k1, :)- X(k2, :), diag_first, 'off'));
            %[M(k1,k2), M(k2,k1)]  = deal(anova1(X(k2, :), diag_first, 'off'));
        end
    end
    
	disp(field_i);
    M
    disp('******************************');
    disp(' ');
    
end
