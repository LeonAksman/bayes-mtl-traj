%*********************************
function vec_out    = repVec(vec_in, ni)

assert(length(vec_in) == length(ni));

vec_out             = zeros(sum(ni), 1);
count               = 1;
for i = 1:length(ni)
    index_i          = count:(count + ni(i) - 1);
    count            = count + ni(i);
    vec_out(index_i) = repmat(vec_in(i), ni(i), 1);
end
