function [designMat_cell_out, stats_design]  = normalizeDesignMat(designMat_cell, P, stats_design_in)

designMat_vert                          = vertcat(designMat_cell{:});
assert(unique(designMat_vert(:, 1)) == 1);

%calc or set the mean/variance
if nargin < 3
    [stats_design.mean, stats_design.std]   = deal(mean(designMat_vert), std(designMat_vert));
else
    stats_design                        = stats_design_in;
end

%standardize
for i = 2:(P+1)
    designMat_vert(:, i)             	= (designMat_vert(:, i) - stats_design.mean(i)) ./ stats_design.std(i);
end

%put it back into a cell array
designMat_cell_out                      = designMat_cell;
iTrain                                  = 1;
for i = 1:size(designMat_cell,1)
    nSamples_i                          = size(designMat_cell{i}, 1);
    
    designMat_cell_out{i}             	= designMat_vert((iTrain + (1:nSamples_i) - 1), :);
    iTrain                              = iTrain + nSamples_i;
end

