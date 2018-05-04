%***********************
function [eta_squared, p]          	= computeEtaSquared(biomarker, diagnoses)

[p,     tbl]                        = anova1(biomarker,	diagnoses, 'off');
eta_squared                         = tbl{2,2}/tbl{4,2};