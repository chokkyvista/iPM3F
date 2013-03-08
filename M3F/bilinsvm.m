function [U,V,theta,fobj,etime] = bilinsvm(U, V, theta, Su, Sv, L, N, M, K, C, l, T_ris, T_rjs, loopi, wors)
etime = 0;

reps = 0.6687^floor(0.1*(loopi-1));
% sR = 2^floor(0.1*(loopi-1));
%% U | V, theta
tStart = tic;
if ~wors
    nfobj = 0.5*(norm(V, 'fro')^2);
else
    nfobj = 0;
end
parfor i = 1:N
    T_rj = T_rjs{i};
    if isempty(T_rj)
        U(i,:) = 0;
    else
        theta_rj = repmat(theta(i,:)', 1, numel(Su{i}));
        Vt = V(Su{i},:)';
        tU = U(i,:);
        tfobju = fObj_U(tU, theta_rj, T_rj, Vt, L, C, l);
        
        bilinear_svm_opt(T_rj, tU, Vt, theta_rj, C, l, reps, tfobju);
%         b_rj = (2*double(T_rj)-1).*theta_rj;
%         % Naive Coordinate Descent
%         bcd_search_shpcr(T_rj, Vt, b_rj, C, l, 1, sR, 0, K, tU);
%         % Powell Method
%         d = [eye(K), zeros(K,1)];
%         bcd_search_shpcr(T_rj, Vt, b_rj, C, l, 1, sR, 1, d, tU);
%         % Rosenbrock Method (4.8t)
%         d = eye(K); slen = zeros(1,K);
%         for n = 1:sR
%             tU = tU*d; bcd_search_shpcr(T_rj, d'*Vt, b_rj, C, l, 1, 1, 2, slen, tU);
%             tU = tU*d'; d = d*rb_dset(eye(K), slen);
%         end
%         % Rosenbrock Method (slen) (5.2t)
%         d = [eye(K), zeros(K,1)];
%         for n = 1:sR
%             bcd_search_shpcr(T_rj, Vt, b_rj, C, l, 1, 1, 2, d, tU);
%             d(:,1:K) = rb_dset(d(:,1:K),d(:,end)');
%         end
%         % Rosenbrock Method (slen, no-mex)
%         d = eye(K);
%         T_rj = double(T_rj);
%         T_rj(T_rj~=1) = -1;
%         tzcrs = T_rj.*(theta_rj-repmat(tU*Vt,L-1,1)) - l;
%         for k = 1:K
%             slope = reshape(T_rj.*repmat(d(:,k)'*Vt,L-1,1), 1, []);
%             zcrs = tzcrs(:)'./slope;
%             slen = search_shpcr(zcrs, slope, C, [-inf,(d(:,k)'*d(:,k))*zcrs+tU*d(:,k)], @(x)(x-tU*d(:,k))/(d(:,k)'*d(:,k)));
%             dU = slen*d(:,k)';
%             tU = tU + dU;
%             tzcrs = tzcrs - slen*reshape(slope, size(tzcrs));
%         end
        
        U(i,:) = tU;
        nfobju = fObj_U(tU, theta_rj, T_rj, Vt, L, C, l);
        nfobj = nfobj + nfobju;
%         if nfobju > tfobju
%             error('C = %.4f, i = %d: %.4f -> %.4f\n', C, i, tfobju, nfobju);
%         end
%         U(i,:) = bilinear_svm_opt(T_rj, tU, Vt, theta_rj, C, l, reps, tfobju);
    end
end
tElapsed = toc(tStart);
fprintf('%.4f: %.2f (%.2fs) | U\n', C, nfobj, tElapsed);
% fprintf('%.4f: %.2f (%.2fs) | U\n', l, nfobj, tElapsed);
etime = etime + tElapsed;

%% V | U, theta
if ~wors
    tStart = tic;
    nfobj = 0.5*(norm(U, 'fro')^2);
    parfor j = 1:M
        T_ri = T_ris{j};
        if isempty(T_ri)
            V(j,:) = 0;
        else
            theta_ri = theta(Sv{j},:)';
            Ut = U(Sv{j},:)';
            tV = V(j,:);
            tfobjv = fObj_V(tV, theta_ri, T_ri, Ut, L, C, l);
            
            bilinear_svm_opt(T_ri, tV, Ut, theta_ri, C, l, reps, tfobjv);
%             b_ri = (2*double(T_ri)-1).*theta_ri;
%             % Naive Coordinate Descent
%             bcd_search_shpcr(T_ri, Ut, b_ri, C, l, 1, sR, 0, K, tV);
%             % Powell Method
%             d = [eye(K), zeros(K,1)];
%             bcd_search_shpcr(T_ri, Ut, b_ri, C, l, 1, sR, 1, d, tV);
%             % Rosenbrock Method
%             d = eye(K); slen = zeros(1,K);
%             for n = 1:sR
%                 tV = tV*d; bcd_search_shpcr(T_ri, d'*Ut, b_ri, C, l, 1, 1, 2, slen, tV);
%                 tV = tV*d'; d = d*rb_dset(eye(K), slen);
%             end
%             % Rosenbrock Method (no-mex)
%             T_ri = double(T_ri);
%             T_ri(T_ri~=1) = -1;
%             tzcrs = T_ri.*(theta_ri-repmat(tV*Ut,L-1,1)) - l;
%             for k = 1:K
%                 slope = reshape(T_ri.*Ut(k*ones(1,L-1),:), 1, []);
% %                 zcrs = reshape(T_ri.*(theta_ri-repmat(tV*Ut-tV(k)*Ut(k,:),L-1,1))-l, 1, [])./slope;
%                 tzcrs = tzcrs + tV(k)*reshape(slope, size(tzcrs));
%                 zcrs = tzcrs(:)'./slope;
%                 tV(k) = search_shpcr(zcrs, slope, C);
%                 tzcrs = tzcrs - tV(k)*reshape(slope, size(tzcrs));
%             end
            
            V(j,:) = tV;
            nfobjv = fObj_V(tV, theta_ri, T_ri, Ut, L, C, l);
            nfobj = nfobj + nfobjv;
%             if nfobjv > tfobjv
%                 error('C = %.4f, j = %d: %.4f -> %.4f\n', C, j, tfobjv, nfobjv);            
%             end
%             V(j,:) = bilinear_svm_opt(T_ri, tV, Ut, theta_ri, C, l, reps, tfobjv);
        end
    end
    tElapsed = toc(tStart);
    fprintf('%.4f: %.2f (%.2fs) | V\n', C, nfobj, tElapsed);
    % fprintf('%.4f: %.2f (%.2fs) | V\n', l, nfobj, tElapsed);
    etime = etime + tElapsed;
end

%% theta | U, V
tStart = tic;
x1 = zeros(N, L-1);
x2 = zeros(N, L-1);
for i = 1:N
    T_rj = T_rjs{i};
    UV = U(i,:)*V(Su{i},:)';
    if ~isempty(UV)
        for r = 1:L-1
            deszcrs = l+UV(T_rj(r,:));
            aszcrs = UV(~T_rj(r,:))-l;
            [x1(i,r), x2(i,r)] = search_theta(deszcrs, aszcrs);
%             [x1(i,r), x2(i,r)] = search_shpcr([deszcrs, aszcrs], [-ones(1,numel(deszcrs)),ones(1,numel(aszcrs))]);
        end
    end
end
theta = (x1+x2)/2;
ftheta = theta(isfinite(theta));
theta(theta==+inf) = max(ftheta) + l;
theta(theta==-inf) = min(ftheta) - l;
tElapsed = toc(tStart);
fobj = fObj(U, V, theta, Su, L, N, C, l, T_rjs, wors);
fprintf('%.4f: %.2f (%.2fs) | theta\n', C, fobj, tElapsed);
% fprintf('%.4f: %.2f (%.2fs) | theta\n', l, fobj, tElapsed);
etime = etime + tElapsed;

end

%%
% options.verbose = 1;
% options.maxIter = 1500;
% for i = 1:N
%     Tjr = T_rjs{i};
%     Tjr(Tjr~=1) = -1;
%     A = [diag(-Tjr(:))*repmat(V(Su{i},:), L-1, 1), eye(numel(Su{i})*(L-1))];
%     A = sparse([A; [zeros(numel(Su{i})*(L-1), p), eye(numel(Su{i})*(L-1))]]);
%     b = diag(Tjr(:))*reshape(repmat(theta(i,:), numel(Su{i}), 1), [], 1)-1;
%     b = [b; zeros(numel(Su{i})*(L-1), 1)];
%     funProj = @(x)linearProject(x, A, b);
%     funObj = @(x)UfunObj(x, p, C);
%     U(i,:) = minConf_PQN(funObj, [U(i,:), zeros(1, numel(Su{i})*(L-1))], funProj, options);
%     fprintf('%d\n', i);
% end

% for i = 1:N
%     w = [ones(numel(Su{i}),1); 0];
%     T_rj = double(repmat(Y(i,Su{i}), L-1, 1) <= repmat((1:L-1)', 1, numel(Su{i})));
%     T_rj(T_rj~=1) = -1;
%     UV = U(i,:)*V(Su{i},:)';
%     for r = 1:L-1
%         A = [eye(numel(Su{i})), T_rj(r,:)'];
%         A = [A; [eye(numel(Su{i})), zeros(numel(Su{i}), 1)]];
%         [x,fval,exitflag,output,lambda] = linprog(w, -A, -[UV'.*T_rj(r,:)'+1; zeros(numel(Su{i}), 1)]);
%         theta(i,r) = x(end);
%     end
% end
