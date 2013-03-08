% calculate the objective function value of Gibbs iPM3F,
% which only differs from that of M3F in the regularizer on Z.
% Here we use -log(IBP(Z)) after 'binarizing' Z.
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function [f] = fobj(Z, V, theta, T, Su, C, ell, rho, varsigma, alphav, ijn, wors)
if ~wors
    f = 0.5*(norm(V, 'fro')^2);
else
    f = 0;
end

[N,K] = size(Z);
logpibpZ = K*log(alphav) - K*sum(log(1:N)); % - alphav*hrmsum(N);
feahist = containers.Map('KeyType', 'char', 'ValueType', 'double');
for k = 1:K
    bZk = Z(:,k) > 0.5;
    if any(bZk)
        histr = sprintf('%d', bZk);
        if feahist.isKey(histr)
            feahist(histr) = feahist(histr) + 1;
        else
            feahist(histr) = 1;
        end
    end
    mk = round(sum(Z(:,k)));
    logpibpZ = logpibpZ + sum(log(1:N-mk)) + sum(log(1:mk-1));
end
logpibpZ = logpibpZ - sum(cellfun(@(x)sum(log(1:x)), feahist.values));
f = f - logpibpZ;

L = size(theta,2)+1;
for i = 1:N
    theta_jr = repmat(theta(i,:), numel(Su{i}), 1);
    xi = max(ell-T(ijn(i,Su{i}),:).*(theta_jr-repmat(V(Su{i},:)*Z(i,:)',1,L-1)), 0);
    f = f + C*sum(xi(:));
    f = f + 0.5*sum((theta(i,:)-rho).^2)./varsigma^2;
end

end
