% More efficient and numerically-stable closed-form solutions
% for 'inv(X_k)' in the Semi-Collapsed GIBBS sampler
% 
% X_k = (a-b)*I_{k*k} + b*1_{k*k}
% e.g. [[a b b b];
%       [b a b b];
%       [b b a b];
%       [b b b a]];
% 
% dets:   dets(k) = det(X_k), output by scgibbshelper
% suminv: suminv(k) = sum(sum(inv(X_k))), output by scgibbshelper
% newk:   new K (number of features) as chosen by the sampler
% Sigma:  Sigma = inv(X_newk)
% 
% See also scgibbshelper
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function Sigma = scgibbshelperinv(dets, suminv, newk)
if newk == 1
    Sigma = shiftdim(suminv(:,1)',-1);
    return;
end
inva = dets(:,newk-1)./dets(:,newk);
if newk == 2
    invb = -inva+0.5*dets(:,2).*suminv(:,2)./dets(:,newk);
else
    invb = -inva+0.5*dets(:,2).*suminv(:,2).*dets(:,newk-2)./dets(:,newk);
end
Sigma = repmat(shiftdim(invb',-1), [newk, newk, 1]);
for kk = 0:newk-1
    Sigma(kk*(newk+1)+1:newk*newk:end) = inva';
end

end