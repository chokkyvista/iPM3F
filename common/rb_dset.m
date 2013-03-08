% calculate the new direction set according to the Rosenbrock method
%
% Written by Minjie Xu (chokkyvista06@gmail.com)

function [nd] = rb_dset(d, slen) % slen: row vector
K = numel(slen);
nd = cumsum(d(:,end:-1:1).*slen(ones(1,size(d,1)),end:-1:1), 2);
nd = nd(:,end:-1:1);
eps = 1e-5;
nd(:,slen<eps) = d(:,slen<eps);
diagind = 1:K+1:K^2;
while rank(nd) < K && eps < 1e-2
    nd(diagind) = nd(diagind) + eps;
    eps = eps * 2;
end
assert(eps < 1e-2);
nd = gs_orthog(nd);
end