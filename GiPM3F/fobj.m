% calculate the objective function value of Gibbs iPM3F,
% which only differs from that of M3F in the regularizer on Z.
% Here we use -log(IBP(Z)) after 'binarizing' Z.
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function [f] = fobj(Z, V, theta, T, Su, C, ell, rho, varsigma, alphav, sigmav, ijn, wors)
if ~wors
    f = (0.5/sigmav^2)*(norm(V, 'fro')^2);
else
    f = 0;
end

f = f - logpibp(Z, alphav);

L = size(theta,2)+1;
for i = 1:size(Z,1)
    theta_jr = repmat(theta(i,:), numel(Su{i}), 1);
    xi = max(ell-T(ijn(i,Su{i}),:).*(theta_jr-repmat(V(Su{i},:)*Z(i,:)',1,L-1)), 0);
    f = f + C*sum(xi(:));
    f = f + 0.5*sum((theta(i,:)-rho).^2)./varsigma^2;
end

end
