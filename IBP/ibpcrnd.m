% sample from the conditional Indian Buffet Process
% (one iteration of Gibbs update)
% 
% inputs:
% Z - sample to be conditioned on
% alpha - as is
% iord - a permutation of row indices
% 
% outputs:
% Z - updated sample
% 
% Note: Samples can be lof-equivalent to each other in this case.
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function Z = ibpcrnd(Z, alpha, iord)
[N, K] = size(Z);
m = sum(Z);
for i = iord
    m = m - Z(i,:);
    Z(i,:) = double(rand(1,K) <= m/N);
    m = m + Z(i,:);
    newk = poissrnd(alpha/N);
    Z(i,K+1:K+newk) = 1;
    m(K+1:K+newk) = 1;
    K = K + newk;
end
Z = Z(:,sum(Z)~=0);

end