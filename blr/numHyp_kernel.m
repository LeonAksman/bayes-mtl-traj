function n = numHyp_kernel(kernelStruct)

%count the linear mixing hypers
n           = length(kernelStruct);

for i = 1:length(kernelStruct)
    switch kernelStruct(i).type
        case 'linear'
            continue;
        case 'gaussian'
            n = n + 1;
        case 'outlier'
            continue;
        otherwise
            error('unknown kernel type: %s', kernelStruct(i).type);
    end
end