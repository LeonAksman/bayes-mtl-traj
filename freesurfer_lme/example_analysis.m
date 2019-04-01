function example_analysis()

addpath(genpath('./lme'));

load './data/univariate/ADNI791_Hipp_and_Entorh.mat';

total_hipp_vol_stats = lme_fit_FS(X_Hipp,[1 2],Y(:,1)+Y(:,2),ni);

total_hipp_vol_stats_1RF = lme_fit_FS(X_Hipp,[1],Y(:,1)+Y(:,2),ni);


lr = lme_LR(total_hipp_vol_stats.lreml,total_hipp_vol_stats_1RF.lreml,1);

C = [zeros(3,3) [1 0 0 0 0; -1 0 1 0 0; 0 0 -1 0 1] zeros(3,6)];

F_C = lme_F(total_hipp_vol_stats,C);

