% calculate the objective function value of iPM3F (differing by a constant)
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function [fobj] = fObj(gammav, psiv, Lambda, theta, alphav, sigmav, Su, L, N, K, C, l, T_rjs, wors)
if ~wors
    fobj = norm(Lambda, 'fro')^2/(2*sigmav^2); % KL(p(V)||p(V0))
else
    fobj = 0;
end
psi_gammav = psi(gammav);
psi_sum_gammav = psi(sum(gammav, 2));
qm_raw_index = psi_gammav(:,2) + cumsum([0;psi_gammav(1:end-1,1)]) - cumsum(psi_sum_gammav);
qm_raw = exp(qm_raw_index);
cumsum_qm_raw = cumsum(qm_raw);
cumsum_psi_gammav1 = cumsum(psi_gammav(:,1));
L_nu = cumsum(qm_raw.*psi_gammav(:,2) + [0;qm_raw(2:end).*cumsum_psi_gammav1(1:end-1)] - qm_raw.*cumsum(psi_sum_gammav) - qm_raw.*qm_raw_index)./cumsum_qm_raw + log(cumsum_qm_raw);
cumsum_psi_sum_gammav = cumsum(psi_sum_gammav);
fobj = fobj + sum((gammav(:,1)-alphav).*(psi_gammav(:,1)-psi_sum_gammav)+(gammav(:,2)-1).*(psi_gammav(:,2)-psi_sum_gammav)-betaln(gammav(:,1),gammav(:,2))) - K*log(alphav); % E[p(\nu,Z)]
fobj = fobj + sum(xlogx(psiv(:))+xlogx(1-psiv(:))) - sum(psiv)*(cumsum_psi_gammav1-cumsum_psi_sum_gammav) - sum(1-psiv)*L_nu; % E[p_0(\nu,Z)]
for i = 1:N
    T_rj = double(T_rjs{i});
    T_rj(T_rj~=1) = -1;
    psivi_Lambda_jkt = psiv(i,:)*Lambda(Su{i},:)';
    [psivi_Lambda_jkt_idx, theta_i_idx] = meshgrid(1:numel(Su{i}), 1:L-1);
    R_rj = max(l-T_rj(:).*(theta(i,theta_i_idx(:))-psivi_Lambda_jkt(psivi_Lambda_jkt_idx(:)))', 0);
    fobj = fobj + C*sum(R_rj(:)); % loss
end

end

function [y] = xlogx(x)
y = x.*log(x+(x==0));
end