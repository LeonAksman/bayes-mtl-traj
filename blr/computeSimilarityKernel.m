function kernel                 = computeSimilarityKernel(inVec)

inVec                           = asRowCol(inVec, 'col');

n                               = length(inVec);
unique_vals                     = unique(inVec);
kernel                          = zeros(n, n);

for i = 1:length(unique_vals)
    is_i                        = double(inVec == unique_vals(i));
    kernel                      = kernel + is_i * is_i';
end

