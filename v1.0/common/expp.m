% 'exp' but for a constant factor to avoid overflow
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function y = expp(x)
y = exp(x-mean(x(:)));

end