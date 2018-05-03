function tasks              = toTasksCell(samplesVec, nSamplesPerTaskVec)

nTasks                      = length(nSamplesPerTaskVec);

tasks                       = cell(nTasks, 1);
iCurr                       = 1;
for i = 1:nTasks
    n_i                     = nSamplesPerTaskVec(i);
       
    tasks{i}                = samplesVec(iCurr:(iCurr + n_i - 1), :);
    iCurr                   = iCurr + n_i;
end
