function [gammav,psiv,Lambda,theta,fobj,etime] = varinfr(gammav, psiv, Lambda, theta, alphav, sigmav, Su, Sv, L, N, M, K, C, l, rho, varsigma, T_ris, T_rjs, loopi, wors)
etime = 0;

if ~wors
    sR = 2^floor(0.1*(loopi-1));
    reps = 0.6687^floor(0.1*(loopi-1));
    %% update Lambda (p(V))
    tStart = tic;
    parfor j = 1:M
        T_ri = T_ris{j};
        if isempty(T_ri)
            Lambda(j,:) = 0;
        else
            theta_ri = theta(Sv{j},:)';
            psivt = psiv(Sv{j},:)';
            tLambda = Lambda(j,:);
            bilinear_svm_opt(T_ri, tLambda, psivt, theta_ri, C*sigmav^2, l, reps);
%             bcd_search_shpcr(T_ri, psivt, (2*double(T_ri)-1).*theta_ri, C, l, 1, sR, 0, K, tLambda);
            Lambda(j,:) = tLambda;
        end
    end
    tElapsed = toc(tStart);
    fprintf('%.4f: %.2f (%.2fs) | Lambda\n', C, fObj(gammav, psiv, Lambda, theta, alphav, sigmav, Su, L, N, K, C, l, T_rjs, wors), tElapsed);
    etime = etime + tElapsed;
end

%% preliminaries
psi_gammav = psi(gammav);
psi_sum_gammav = psi(sum(gammav, 2));

qm_raw_index = psi_gammav(:,2) + cumsum([0;psi_gammav(1:end-1,1)]) - cumsum(psi_sum_gammav);
qm_raw = exp(qm_raw_index);
cumsum_qm_raw = cumsum(qm_raw);

% qmk = repmat(qm_raw, 1, K)./cumsum_qm_raw(:,ones(1,K))';

%% update psiv (p(Z))
tStart = tic;
cumsum_psi_gammav1 = cumsum(psi_gammav(:,1));
cumsum_psi_sum_gammav = cumsum(psi_sum_gammav);
L_nu = cumsum(qm_raw.*psi_gammav(:,2) + [0;qm_raw(2:end).*cumsum_psi_gammav1(1:end-1)] - qm_raw.*cumsum_psi_sum_gammav - qm_raw.*qm_raw_index)./cumsum_qm_raw + log(cumsum_qm_raw);

% for i = 1:N
%     T_rj = double(T_rjs{i});
%     T_rj(T_rj~=1) = -1;
%     T_rj(T_rj.*(repmat(theta(i,:)', 1, numel(Su{i}))-repmat(psiv(i,:)*Lambda(Su{i},:)', L-1, 1)) > l) = 0;
%     partial_Rli = (sum(T_rj)*Lambda(Su{i},:))';
%     psiv(i,:) = 1./(1+exp(-(cumsum_psi_gammav1-cumsum_psi_sum_gammav-L_nu-C*partial_Rli))');
% end

sR = 2^floor(0.1*(loopi-1));
%     function y = fObj_psivik(x)
%         new_Lambda_jk_psivit = Lambda_jk_psivit + (x-psivik)*Lambda_jk(:,k);
%         y = x*log(x)+(1-x)*log(1-x)...
%             -(cumsum_psi_gammav1(k)-cumsum_psi_sum_gammav(k))*x-L_nu(k)*(1-x)...
%             +C*sum(max(l-T_rj(:).*(theta_rj(:)-new_Lambda_jk_psivit(Lambda_jk_psivit_idx(:))),0));
%     end
% for i = 1:N
%     T_rj = double(T_rjs{i});
%     T_rj(T_rj~=1) = -1;
%     [Lambda_jk_psivit_idx, theta_i_idx] = meshgrid(1:numel(Su{i}), 1:L-1);
%     theta_rj = theta(i, theta_i_idx(:));
%     Lambda_jk = Lambda(Su{i},:);
%     Lambda_jk_psivit = Lambda_jk*psiv(i,:)';
%     for r = 1:sR
%         for k = 1:K
%             psivik = psiv(i,k);
%             psiv(i,k) = golden(@fObj_psivik, 0, 1, 10);
%             Lambda_jk_psivit = Lambda_jk_psivit + (psiv(i,k)-psivik)*Lambda_jk(:,k);
%         end
%     end
% end

parfor i = 1:N
    T_rj = T_rjs{i};
    Lambda_jk = Lambda(Su{i},:);
    psivi = psiv(i,:);
    thetai_r = theta(i,:);
%     gs_search_psivi(T_rj, thetai_r, Lambda_jk, cumsum_psi_gammav1-cumsum_psi_sum_gammav, L_nu, C, l, Lambda_jk*psivi', psivi, sR);
    bcd_search_shpcr(T_rj, Lambda_jk', (2*double(T_rj)-1).*thetai_r(ones(1,numel(Su{i})),:)', C, l, 2, sR, 0, K, psivi, cumsum_psi_sum_gammav-cumsum_psi_gammav1, -L_nu);
    psiv(i,:) = psivi;
end

tElapsed = toc(tStart);
fprintf('%.4f: %.2f (%.2fs) | psi\n', C, fObj(gammav, psiv, Lambda, theta, alphav, sigmav, Su, L, N, K, C, l, T_rjs, wors), tElapsed);
etime = etime + tElapsed;

%% update gammav (p(\nu))
tStart = tic;
sum_psivk = sum(psiv)';
gammav(end:-1:1,2) = 1 + cumsum((N-sum_psivk(end:-1:1))./cumsum_qm_raw(end:-1:1)).*qm_raw(end:-1:1);
gammav(end:-1:1,1) = alphav + cumsum(sum_psivk(end:-1:1)) + cumsum([0;gammav(end:-1:2,2)-1]);
tElapsed = toc(tStart);
fprintf('%.4f: %.2f (%.2fs) | gamma\n', C, fObj(gammav, psiv, Lambda, theta, alphav, sigmav, Su, L, N, K, C, l, T_rjs, wors), tElapsed);
etime = etime + tElapsed;

%% update theta
tStart = tic;
x1 = zeros(N, L-1);
x2 = zeros(N, L-1);
for i = 1:N
    T_rj = T_rjs{i};
    ZV = psiv(i,:)*Lambda(Su{i},:)';
    if ~isempty(ZV)
        for r = 1:L-1
            deszcrs = l+ZV(T_rj(r,:));
            aszcrs = ZV(~T_rj(r,:))-l;
%             [x1(i,r), x2(i,r)] = search_theta(deszcrs, aszcrs, C, rho(r), varsigma);
            zcrs = [deszcrs, aszcrs];
            [x1(i,r), x2(i,r)] = search_shpcr(zcrs, [-ones(1,numel(deszcrs)),ones(1,numel(aszcrs))], C, ([-inf,zcrs]-rho(r))./varsigma^2, @(x)varsigma^2*x+rho(r));
        end
    end
end
theta = (x1+x2)/2;
tElapsed = toc(tStart);
etime = etime + tElapsed;

%% calculate fobj
fobj = fObj(gammav, psiv, Lambda, theta, alphav, sigmav, Su, L, N, K, C, l, T_rjs, wors);
fprintf('%.4f: %.2f (%.2fs) | theta\n', C, fobj, tElapsed);

end