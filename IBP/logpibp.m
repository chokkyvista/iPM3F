% calculate log(pIBP([Z]))
% 
% Note:
% 1. the probability is calculated w.r.t. the lof-equivalent class [Z]
% 2. when Z is real-valued (e.g. averaged over multiple binary samples),
%    it is first discretized to a binary matrix
% 
% See also GAMMALN
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function logpibpZ = logpibp(Z, alphav)
Z = Z > 0.5;
Z = Z(:,sum(Z)>0);

[N,K] = size(Z);
logpibpZ = K*log(alphav) - K*gammaln(N+1) - alphav*hrmsum(N);
feahist = containers.Map('KeyType', 'char', 'ValueType', 'double');
for k = 1:K
    histr = sprintf('%d', Z(:,k));
    if feahist.isKey(histr)
        feahist(histr) = feahist(histr) + 1;
    else
        feahist(histr) = 1;
    end
    mk = sum(Z(:,k));
    logpibpZ = logpibpZ + gammaln(N-mk+1) + gammaln(mk);
end
logpibpZ = logpibpZ - sum(cellfun(@(x)gammaln(x+1), feahist.values));

end