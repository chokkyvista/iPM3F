% Golden-section search
% 
% N - number of search steps
%
% Written by Minjie Xu (chokkyvista06@gmail.com)

function x = golden(fun, a0, b0, N)
r = 0.381966011250105;
a = a0; b = b0;
s = a + r*(b-a);
t = a+b - s;
f1 = fun(s);
f2 = fun(t);
for n = 1:N-1
    if f1 < f2
        b = t;
        t = s;
        f2 = f1;
        s = a + b - t;
        f1 = fun(s);
    else
        a = s;
        s = t;
        f1 = f2;
        t = a + b - s;
        f2 = fun(t);
    end
end
if f1 < f2
    x = s;
else
    x = t;
end

end