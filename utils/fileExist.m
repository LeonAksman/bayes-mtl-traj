function bExist = fileExist(filenames)

if iscell(filenames)
    bExist          = zeros(size(filenames));

    for i = 1:length(filenames)        
        bExist(i)   = (exist(filenames{i}, 'file') == 2);
    end
elseif ischar(filenames)
    bExist  = (exist(filenames, 'file') == 2);
else
    error('fileExist looks for cells or chars');
end
