function [vars_opt, fobj, etime] = varinfr(vars_opt, vars_init, alphav, Su, Sv, L, N, M, K, l, rho, varsigma, T_ris, T_rjs, C, loopi, wors)
Lambda = vars_opt.Lambda;
psiv = vars_opt.psi;
gammav = vars_opt.gamma;
theta = vars_opt.theta;
etime = 0;

if ~wors
    %% update mu, beta, W, tau (p(\mu,\Omega))
    tStart = tic;
    vars_opt.tau = vars_init.tau + M;
    vars_opt.beta = vars_init.beta + M;
    vars_opt.mu = (vars_init.beta*vars_init.mu+sum(Lambda))./vars_opt.beta;
    vars_opt.invW = vars_init.invW + M*vars_opt.invXi + Lambda'*Lambda + ...
        vars_init.beta*(vars_init.mu'*vars_init.mu) - vars_opt.beta*(vars_opt.mu'*vars_opt.mu);
    tElapsed = toc(tStart);
    fprintf('%.4f: %.2f (%.2fs) | Gaussian-Wishart\n', C, fObj(vars_opt, alphav, Su, L, M, N, K, l, T_rjs, C, vars_init, wors), tElapsed);
    etime = etime + tElapsed;

    reps = 0.6687^floor(0.1*(loopi-1));
    %% update Lambda, Xi (p(V))
    tStart = tic;
    vars_opt.invXi = vars_opt.invW ./ vars_opt.tau;
    invP = chol(vars_opt.invXi);
    mu = vars_opt.mu;
    parfor j = 1:M
        T_ri = T_ris{j};
        if isempty(T_ri)
            Lambda(j,:) = mu;
        else
            theta_ri = theta(Sv{j},:)';
            psivt = psiv(Sv{j},:)';
            tLambda = Lambda(j,:);
            bilinear_svm_opt(T_ri, tLambda, invP*psivt, theta_ri-mu(ones(1,L-1),:)*psivt, C, l, reps);
            Lambda(j,:) = tLambda*invP + mu;
        end
    end
    tElapsed = toc(tStart);
    vars_opt.Lambda = Lambda;
    fprintf('%.4f: %.2f (%.2fs) | Lambda,Xi\n', C, fObj(vars_opt, alphav, Su, L, M, N, K, l, T_rjs, C, vars_init, wors), tElapsed);
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
%     for r = 1:gsR
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
%     gs_search_psivi(T_rjs{i}, theta(i,:), Lambda_jk, cumsum_psi_gammav1-cumsum_psi_sum_gammav, L_nu, C, l, Lambda_jk*psivi', psivi, sR);
    bcd_search_shpcr(T_rj, Lambda_jk', (2*double(T_rj)-1).*thetai_r(ones(1,numel(Su{i})),:)', C, l, 2, sR, 0, K, psivi, cumsum_psi_sum_gammav-cumsum_psi_gammav1, -L_nu);
    psiv(i,:) = psivi;
end

tElapsed = toc(tStart);
vars_opt.psi = psiv;
fprintf('%.4f: %.2f (%.2fs) | psi\n', C, fObj(vars_opt, alphav, Su, L, M, N, K, l, T_rjs, C, vars_init, wors), tElapsed);
etime = etime + tElapsed;

%% update gammav (p(\nu))
tStart = tic;
sum_psivk = sum(psiv)';
gammav(end:-1:1,2) = 1 + cumsum((N-sum_psivk(end:-1:1))./cumsum_qm_raw(end:-1:1)).*qm_raw(end:-1:1);
gammav(end:-1:1,1) = alphav + cumsum(sum_psivk(end:-1:1)) + cumsum([0;gammav(end:-1:2,2)-1]);
tElapsed = toc(tStart);
vars_opt.gamma = gammav;
fprintf('%.4f: %.2f (%.2fs) | gamma\n', C, fObj(vars_opt, alphav, Su, L, M, N, K, l, T_rjs, C, vars_init, wors), tElapsed);
etime = etime + tElapsed;

%% update theta
tStart = tic;
for i = 1:N
    x1 = zeros(1, L-1);
    x2 = zeros(1, L-1);
    T_rj = T_rjs{i};
    ZV = psiv(i,:)*Lambda(Su{i},:)';
    if ~isempty(ZV)
        for r = 1:L-1
            [x1(r), x2(r)] = search_theta(l+ZV(T_rj(r,:)), ZV(~T_rj(r,:))-l, C, rho(r), varsigma);
        end
    end
    theta(i,:) = (x1+x2)/2;
end
tElapsed = toc(tStart);
vars_opt.theta = theta;
etime = etime + tElapsed;

%% calculate fobj
fobj = fObj(vars_opt, alphav, Su, L, M, N, K, l, T_rjs, C, vars_init, wors);
fprintf('%.4f: %.2f (%.2fs) | theta\n', C, fobj, tElapsed);

end