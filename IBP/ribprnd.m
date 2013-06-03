% sample from the (non-exchangeable) real-valued extension to the Indian buffet process
% 
% N - number of objects
% alpha - as is
% urnd - the function handle of a random number generator
% unirow - whether each row has identical real values
% Z - the sampled latent feature matrix
% 
% Note: Samples can be lof-equivalent to each other in this case.
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function Z = ribprnd(N, alpha, urnd, unirow)
if ~exist('urnd', 'var')
    urnd = @rand;
end
if ~exist('unirow', 'var')
    unirow = false;
end
EKp = ceil(alpha*hrmsum(N)*10);
Z = zeros(N, EKp);
K = 0;
m = zeros(1, EKp);
for i = 1:N
    if unirow
        ui = nzrnd(urnd, [1,1]);
    else
        ui = nzrnd(urnd, [1,K]);
    end
    Z(i,1:K) = (rand(1,K) <= m(1:K)/i).*ui;
    m(1:K) = m(1:K) + (Z(i,1:K)~=0);
    newk = poissrnd(alpha/i);
    if ~unirow
        ui = nzrnd(urnd, [1,newk]);
    end
    Z(i,K+1:K+newk) = ui;
    m(K+1:K+newk) = 1;
    K = K + newk;
end
Z = Z(:,1:K);

end

% make sure the samples are all non-zeros
function v = nzrnd(urnd, sz)
v = urnd(sz);
zeroidx = v==0;
while (any(zeroidx))
    v(zeroidx) = urnd(1, nnz(zeroidx));
    zeroidx = v==0;
end

end
