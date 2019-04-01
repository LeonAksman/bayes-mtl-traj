%search subset within set, make sure there's exactly one match
function indeces    = stringIndex(stringSubset, stringSet)

len                 = length(stringSubset);

indeces             = zeros(len, 1);
for i = 1:len
    index_i         = find(strcmp(stringSubset{i}, stringSet));
    assert(length(index_i) == 1, 'string: %s, found: %d', stringSubset{i}, length(index_i));
    
    indeces(i)      = index_i;
end