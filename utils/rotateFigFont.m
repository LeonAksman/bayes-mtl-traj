function rotateFigFont(degrees)

objects = get( findobj( gca,'Type','hggroup') ,'Parent' );

if iscell(objects)

    for i = 1:length(objects)
        set(objects{i} ,'XTickLabelRotation', degrees);
    end
else
    %get(findobj( gca,'Type','hggroup') ,'Parent');
    set(objects ,'XTickLabelRotation', degrees);
end