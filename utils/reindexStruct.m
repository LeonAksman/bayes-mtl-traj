function structOut  = reindexStruct(structIn, index, rowCol)

if nargin < 3
    rowCol          = 'row';
end
rowCol              = lower(rowCol);
if strcmp(rowCol, 'row') ~= 1 && strcmp(rowCol, 'col') ~= 1
    error('reindexStruct: rowCol field not set to row or col.');
end

structOut           = structIn;

fields              = fieldnames(structIn);
for i = 1:length(fields)
    
%     if ~isnumeric(structOut.(fields{i}))
%         continue;
%     end
    if isstruct(structOut.(fields{i}))
        continue;
    end
    
    if strcmp(rowCol, 'row') == 1
        structOut.(fields{i})   = structOut.(fields{i})(index, :);
    else
        structOut.(fields{i})   = structOut.(fields{i})(:, index);
    end
end