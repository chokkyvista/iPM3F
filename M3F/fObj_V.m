function [f,xi] = fObj_V(V, theta_ri, T_ri, Ut, L, C, l)
T_ri = double(T_ri);
T_ri(T_ri~=1) = -1;
VU = V*Ut;
xi = max(l-T_ri(:).*(theta_ri(:)-reshape(repmat(VU,L-1,1),[],1)), 0);
f = 0.5*sum(V.^2) + C*sum(xi);
end