function [table_out, outliers, inliers, outlierSubjects]	= filterOutliers(table_in, field_name, THRESHOLD, outliers, inliers)

data                            = table_in.(field_name);
z                               = (data - mean(data))/std(data);

index_bad                       = find(abs(z) > THRESHOLD);
dispf('**** Found %d samples with |z| > %d for measurement: %s', length(index_bad), THRESHOLD, field_name);
outliers.(field_name)           = zeros(length(index_bad), 1);
for i = 1:length(index_bad)
    index_i                     = index_bad(i);
    dispf('subject: %s, image id: %d, z score: %.1f, diagnosis: %s', table_in.SubjectID{index_i}, table_in.id(index_i), z(index_i), table_in.DXGroup{index_i});
    
    outliers.(field_name)(i) = table_in.id(index_i);
end
outlierSubjects                 = table_in.SubjectID(index_bad);

%******* inliers
index_good                      = find(abs(z) <= THRESHOLD);
inliers.(field_name).id      	= zeros(length(index_good), 1);
inliers.(field_name).z      	= zeros(length(index_good), 1);
for i = 1:length(index_good)
    index_i                     = index_good(i);
    
    inliers.(field_name).id(i)  = table_in.id(index_i);
    inliers.(field_name).z(i)   = z(index_i);
end

table_out                    	= table_in(index_good, :);