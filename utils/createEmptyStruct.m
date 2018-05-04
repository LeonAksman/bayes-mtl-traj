function l                  = createEmptyStruct(fields)

l                           = struct;
for i = 1:length(fields)
    l.(fields{i})           = 0;
end