function [f,xi] = fObj_U(U, theta_rj, T_rj, Vt, L, C, l)
T_rj = double(T_rj);
T_rj(T_rj~=1) = -1;
UV = U*Vt;
xi = max(l-T_rj(:).*(theta_rj(:)-reshape(repmat(UV,L-1,1),[],1)), 0);
f = 0.5*sum(U.^2) + C*sum(xi);
end
