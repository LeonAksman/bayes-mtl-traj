function c = conditional(condition, a, b)

assert(length(condition) == 1, 'scalar conditions only for now');

if condition
    c = a;
else
    c = b;
end
