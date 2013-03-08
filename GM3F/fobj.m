% calculate the objective function value of M3F
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function [f] = fobj(U, V, theta, T, Su, C, ell, rho, varsigma, ijn, wors)
if ~wors
    f = 0.5*(norm(V, 'fro')^2);
else
    f = 0;
end
f = f + 0.5*(norm(U, 'fro')^2);
N = size(U,1);
L = size(theta,2)+1;
for i = 1:N
    theta_jr = repmat(theta(i,:), numel(Su{i}), 1);
    xi = max(ell-T(ijn(i,Su{i}),:).*(theta_jr-repmat(V(Su{i},:)*U(i,:)',1,L-1)), 0);
    f = f + C*sum(xi(:));
    f = f + 0.5*sum((theta(i,:)-rho).^2)./varsigma^2;
end

end