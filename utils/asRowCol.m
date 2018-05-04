function out = asRowCol(x, style)


is_row  = isrow(x);
is_col  = iscolumn(x);

assert(is_row || is_col);
assert(strcmp(style, 'row') || strcmp(style, 'col'));

if      is_row
    
    if strcmp(style, 'col')
        out = x';
    else
        out = x;
    end
    
elseif  is_col
    
 	if strcmp(style, 'row')
        out = x';
   	else
        out = x;
    end
    
end
    