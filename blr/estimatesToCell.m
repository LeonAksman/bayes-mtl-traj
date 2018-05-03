function estimatesCell      = estimatesToCell(estimatesVec, nSamplesPerTaskVec)

nTasks                      = length(nSamplesPerTaskVec);

estimatesCell               = cell(nTasks, 1);
iCurr                       = 1;
for i = 1:nTasks
    n_i                     = nSamplesPerTaskVec(i);
    estimatesCell{i}        = estimatesVec(iCurr:(iCurr + n_i - 1));
    iCurr                   = iCurr + n_i;
end
