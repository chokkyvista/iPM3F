function [f] = fObj(U, V, theta, Su, L, N, C, l, T_rjs, wors)
if ~wors
    f = 0.5*(norm(V, 'fro')^2);
else
    f = 0;
end
for i = 1:N
    T_rj = T_rjs{i};
    theta_rj = repmat(theta(i,:)', 1, numel(Su{i}));
    Vt = V(Su{i},:)';
    f = f + fObj_U(U(i,:), theta_rj, T_rj, Vt, L, C, l);
end
end
