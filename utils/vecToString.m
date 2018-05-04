function vecStr = vecToString(format, vec)
%format: e.g. %.1f

vec = asRowCol(vec, 'row');

%sprintf('%.1f ', out.t'))
vecStr = sprintf([format ' '], vec);
