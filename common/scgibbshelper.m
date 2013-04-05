% More efficient and numerically-stable closed-form solutions
% for 2 key matrix manipulations in the Semi-Collapsed GIBBS sampler
% 
% X_k = (a-b)*I_{k*k} + b*1_{k*k}
% e.g. [[a b b b];
%       [b a b b];
%       [b b a b];
%       [b b b a]];
% 
% a, b:   matrices of same dimensiona
% dets:   dets(k) = det(X_k)
% suminv: suminv(k) = sum(sum(inv(X_k)))
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function [dets, suminv] = scgibbshelper(a,b,K)
delta = a - b;
if isvector(a) && isvector(b)
    a = a(:); b = b(:); delta = delta(:);
    dets = zeros(numel(a), K);
    suminv = zeros(size(dets));
    dets(:,1:2) = [a, (a+b).*delta];
    suminv(:,1:2) = [1./a, 2./(a+b)];

    deltapkm1 = delta;
    for k = 3:K
        deltapkm1 = deltapkm1.*delta;
        tmp = k.*b + delta;
        dets(:,k) = deltapkm1.*tmp;
        suminv(:,k) = k./tmp;
    end
else
    dets = zeros([size(a), K]);
    suminv = zeros(size(dets));
    dspan = cell(1, ndims(a));
    dspan(:) = {':'};
    dets(dspan{:},1:2) = [a, (a+b).*delta];
    suminv(dspan{:},1:2) = [1./a, 2./(a+b)];

    deltapkm1 = delta;
    for k = 3:K
        deltapkm1 = deltapkm1.*delta;
        tmp = k.*b + delta;
        dets(dspan{:},k) = deltapkm1.*tmp;
        suminv(dspan{:},k) = k./tmp;
    end
end

end