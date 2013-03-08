% sample from the (non-exchangeable) Indian Buffet Process
% 
% N - number of customers
% alpha - as is
% Z - the sampled binary feature matrix
% 
% Note: Samples can be lof-equivalent to each other in this case.
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function Z = ibprnd(N, alpha)
EKp = ceil(alpha*hrmsum(N)*10);
Z = zeros(N, EKp);
K = 0;
m = zeros(1, EKp);
for i = 1:N
    Z(i,1:K) = double(rand(1,K) <= m(1:K)/i);
    m(1:K) = m(1:K) + Z(i,1:K);
    newk = poissrnd(alpha/i);
    Z(i,K+1:K+newk) = 1;
    m(K+1:K+newk) = 1;
    K = K + newk;
end
Z = Z(:,1:K);

end
