% one round of Gibbs sampling in Gibbs iPM3F
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function [Z,V,theta,invlambda,zcidx,newK,etime,fval] = mcmc(Z, V, theta, invlambda, alphav, sigmav, T, Su, Sv, C, ell, rho, varsigma, ijn, poisstrunc, algtype, loopi, wors)
[N,K] = size(Z);
[M,tK] = size(V);
if K ~= tK
    error('Dimensions of Z & V must agree.');
end
L = size(theta, 2)+1;

if ~any(algtype == 0:2)
    error('cfg:invalidparam', ...
        ['Unknown ''algtype'': %d\n', ...
         '    0 for ''fully uncollapsed Gibbs sampling'';\n', ...
         '    1 for ''semi-collapsed Gibbs sampling'';\n', ...
         '    2 for ''semi-collapsed EM-style''.'], algtype);
end

etime = 0;
for e = 1:1
    % sampling invlambda
    tStart = tic;
    for j = 1:M
        n = ijn(Sv{j},j);
        if isempty(n)
            continue;
        end
        tmu = (2/C)./abs(ell-T(n,:).*(theta(Sv{j},:)-repmat(Z(Sv{j},:)*V(j,:)', 1, L-1)));
        indinf = isinf(tmu);
        if any(indinf(:))
            tmu(indinf) = max(tmu(~indinf)).^2; % avoid infinite mu
        end
        if algtype == 2
            invlambda(n,:) = tmu;
        else
            invlambda(n,:) = reshape(invnrnd(tmu(:), 1, numel(n)*(L-1)), numel(n), []);
        end
    end
    tElapsed = toc(tStart);
    fprintf('%.4f[%d]: %s (%.2fs) | invlambda\n', C, loopi, ...
        num2str([mean(invlambda(:)),std(invlambda(:),1)], ['%.2f',char(177),'%.2f']), tElapsed);
    etime = etime + tElapsed;
    
    % sampling Z
    tStart = tic;
    newks = zeros(1,poisstrunc+1);
    m = sum(Z);
    for i = randperm(N)
        n = ijn(i, Su{i});
        m = m - Z(i,:);
        cv1n = 0.5*C*sum(T(n,:),2)+0.25*C^2*sum((ell*T(n,:)-theta(i*ones(1,numel(n)),:)).*invlambda(n,:),2);
        cv2n = 0.25*C^2*sum(invlambda(n,:),2);
        tV = V(Su{i},:);
        VZit = tV*Z(i,:)';
%         for k = 1:K
%             oZik = Z(i,k);
%             tVk = tV(:,k);
%             p1bp0 = m(k)/(N-m(k))*exp(-tVk'*(cv1n+cv2n.*(VZit+(0.5-oZik)*tVk)));
%             if algtype == 2
%                 Z(i,k) = p1bp0 > 1;
%             else
%                 Z(i,k) = rand*(1+p1bp0) > 1;
%             end
%             VZit = VZit+(Z(i,k)-oZik)*tVk;
%         end
        zrow = Z(i,:);
        zrow_sampler(zrow, m./(N-m), tV, cv1n, cv2n, VZit, 1, algtype);
        Z(i,:) = zrow;
        m = m + Z(i,:);
        
        cv3n = cv1n + cv2n.*VZit;
        if algtype == 0
            % fully uncollapsed sampling
            Vappend = sigmav*randn(M, poisstrunc);
            vjk = cumsum(Vappend(Su{i},:), 2);
            loglhdk = -sum(vjk.*(cv3n(:,ones(1,poisstrunc))+0.5*(cv2n(:,ones(1,poisstrunc)).*(vjk.^2))));
            pk = [1, cumprod((alphav/N)./(1:poisstrunc))].*expp([0,loglhdk]);
            newk = find(rand*sum(pk) > [0,cumsum(pk)], 1, 'last') - 1;
            Z(i,K+1:K+newk) = 1;
            m(K+1:K+newk) = 1;
            K = K + newk;
            V(:,K+1:K+newk) = Vappend(:,1:newk);
        elseif algtype == 1 || algtype == 2
            % semi-collapsed sampling
            [dets, suminv] = scgibbshelper(1./sigmav^2+cv2n, cv2n, poisstrunc);
            loglhdk = -((1:poisstrunc)*(numel(n)*log(sigmav)) + 0.5*log(prod(dets)) - sum(repmat(0.5*cv3n.^2, 1, poisstrunc).*suminv));
            pk = [1, cumprod((alphav/N)./(1:poisstrunc))].*expp([0,loglhdk]);
            if algtype == 2
                [~, newk] = max(pk);
                newk = newk - 1;
            else % algtype == 1
                newk = find(rand*sum(pk) > [0,cumsum(pk)], 1, 'last') - 1;
            end
            if newk > 0
                Z(i,K+1:K+newk) = 1;
                m(K+1:K+newk) = 1;
                tmu = -cv3n.*suminv(:,newk)./newk;
                tmu = tmu(:, ones(1,newk));
                tSigma = scgibbshelperinv(dets, suminv, newk);
                if algtype == 2
                    V(:,K+1:K+newk) = 0;
                    V(Su{i},K+1:K+newk) = tmu;
                else % algtype == 1
                    V(:,K+1:K+newk) = sigmav*randn(M, newk);
                    V(Su{i},K+1:K+newk) = mvnrnd(tmu, tSigma);
                end
                K = K + newk;
            end
        end
        newks(newk+1) = newks(newk+1) + 1;
    end
    newK = sum(newks(2:end).*(1:poisstrunc));
    zcidx = sum(Z(:,1:K-newK))==0;
    Z(:,zcidx) = [];
    V(:,zcidx) = [];
    K = size(Z,2);
    tElapsed = toc(tStart);
    fval = fobj(Z, V, theta, T, Su, C, ell, rho, varsigma, alphav, sigmav, ijn, wors);
    nfeapu = sum(Z,2);
    fprintf('%.4f[%d]: %d, %s, [%s][%s], %.4f (%.2fs) | Z\n', C, loopi, ...
        K, num2str([mean(nfeapu),std(nfeapu(:),1)], ['%.2f',char(177),'%.2f']), ...
        num2str(newks(2:end)./(sum(newks)-newks(1)), '%.2f '), ...
        num2str(newks./(sum(newks)-cumsum([0,newks(1:end-1)])), '%.2f '), ...
        fval, tElapsed);
    etime = etime + tElapsed;
    
    % sampling V
    if ~wors
        tStart = tic;
        parfor j = 1:M
            n = ijn(Sv{j},j);
            if isempty(n)
                R = diag((1/sigmav)*ones(1,K));
                bj = 0;
            else
                tmp = 0.5*C*sqrt(sum(invlambda(n,:),2));
                tmp = Z(Sv{j},:).*tmp(:,ones(K,1));
                invBj = tmp'*tmp;
                invBj(1:K+1:K*K) = invBj(1:K+1:K*K) + 1/sigmav^2;
                R = choll(invBj);
                bj = -(R\(R'\(Z(Sv{j},:)'*(0.5*C*sum(T(n,:),2)+0.25*C^2*sum((ell*T(n,:)-theta(Sv{j},:)).*invlambda(n,:),2)))));
            end
            if algtype == 2
                V(j,:) = bj';
            else
                V(j,:) = (bj + R\randn(K, 1))';
            end
        end
        tElapsed = toc(tStart);
        fval = fobj(Z, V, theta, T, Su, C, ell, rho, varsigma, alphav, sigmav, ijn, wors);
        fprintf('%.4f[%d]: %s (%.2fs) | V\n', C, loopi, ...
            num2str([mean(V(:)),std(V(:),1),fval], ['%.2f',char(177),'%.2f',', %.4f']), tElapsed);
        etime = etime + tElapsed;
    end
    
    % sampling theta
    tStart = tic;
    for i = 1:N
        n = ijn(i,Su{i});
        invAi = 1/varsigma^2 + 0.25*C^2*sum(invlambda(n,:));
        ai = (rho/varsigma^2+0.5*C*sum(T(n,:))+0.25*C^2*sum((ell*T(n,:)+repmat(V(Su{i},:)*Z(i,:)',1,L-1)).*invlambda(n,:)))./invAi;
        if algtype == 2
            theta(i,:) = ai;
        else
            theta(i,:) = ai + randn(1, L-1)./sqrt(invAi);
        end
    end
    tElapsed = toc(tStart);
    fval = fobj(Z, V, theta, T, Su, C, ell, rho, varsigma, alphav, sigmav, ijn, wors);
    fprintf('%.4f[%d]: [%s], %.4f (%.2fs) | theta\n', C, loopi, ...
        num2str(mean(theta), '%.2f '), fval, tElapsed);
    etime = etime + tElapsed;
end


end