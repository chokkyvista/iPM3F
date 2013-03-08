% one round of Gibbs sampling in Gibbs M3F
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function [U,V,theta,invlambda,etime,fval] = mcmc(U, V, theta, invlambda, T, Su, Sv, C, ell, rho, varsigma, ijn, algtype, loopi, wors)
[N,K] = size(U);
[M,tK] = size(V);
if K ~= tK
    error('Dimensions of U & V must agree.');
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
        tmu = (2/C)./abs(ell-T(n,:).*(theta(Sv{j},:)-repmat(U(Sv{j},:)*V(j,:)', 1, L-1)));
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
    
    % sampling U
    tStart = tic;
    parfor i = 1:N
        n = ijn(i,Su{i});
        if isempty(n)
            R = eye(K);
            bi = 0;
        else
            tmp = 0.5*C*sqrt(sum(invlambda(n,:),2));
            tmp = V(Su{i},:).*tmp(:,ones(K,1));
            invBi = tmp'*tmp;
            invBi(1:K+1:K*K) = invBi(1:K+1:K*K) + 1;
            R = choll(invBi);
            bi = -(R\(R'\(V(Su{i},:)'*(0.5*C*sum(T(n,:),2) + ...
                0.25*C^2*sum((ell*T(n,:)-theta(i*ones(1,numel(Su{i})),:)).*invlambda(n,:),2)))));
        end
        if algtype == 2
            U(i,:) = bi';
        else
            U(i,:) = (bi + R\randn(K, 1))';
        end
    end
    tElapsed = toc(tStart);
    fval = fobj(U, V, theta, T, Su, C, ell, rho, varsigma, ijn, wors);
    fprintf('%.4f[%d]: %s (%.2fs) | U\n', C, loopi, ...
        num2str([mean(U(:)),std(U(:),1),fval], ['%.2f',char(177),'%.2f',', %.4f']), tElapsed);
    etime = etime + tElapsed;
    
    % sampling V
    if ~wors
        tStart = tic;
        parfor j = 1:M
            n = ijn(Sv{j},j);
            if isempty(n)
                R = eye(K);
                bj = 0;
            else
                tmp = 0.5*C*sqrt(sum(invlambda(n,:),2));
                tmp = U(Sv{j},:).*tmp(:,ones(K,1));
                invBj = tmp'*tmp;
                invBj(1:K+1:K*K) = invBj(1:K+1:K*K) + 1;
                R = choll(invBj);
                bj = -(R\(R'\(U(Sv{j},:)'*(0.5*C*sum(T(n,:),2) + ...
                    0.25*C^2*sum((ell*T(n,:)-theta(Sv{j},:)).*invlambda(n,:),2)))));
            end
            if algtype == 2
                V(j,:) = bj';
            else
                V(j,:) = (bj + R\randn(K, 1))';
            end
        end
        tElapsed = toc(tStart);
        fval = fobj(U, V, theta, T, Su, C, ell, rho, varsigma, ijn, wors);
        fprintf('%.4f[%d]: %s (%.2fs) | V\n', C, loopi, ...
            num2str([mean(U(:)),std(U(:),1),fval], ['%.2f',char(177),'%.2f',', %.4f']), tElapsed);
        etime = etime + tElapsed;
    end
    
    % sampling theta
    tStart = tic;
    for i = 1:N
        n = ijn(i,Su{i});
        invAi = 1/varsigma^2 + 0.25*C^2*sum(invlambda(n,:));
        ai = (rho/varsigma^2+0.5*C*sum(T(n,:))+0.25*C^2*sum((ell*T(n,:)+repmat(V(Su{i},:)*U(i,:)',1,L-1)).*invlambda(n,:)))./invAi;
        if algtype == 2
            theta(i,:) = ai;
        else
            theta(i,:) = ai + randn(1, L-1)./sqrt(invAi);
        end
    end
    tElapsed = toc(tStart);
    fval = fobj(U, V, theta, T, Su, C, ell, rho, varsigma, ijn, wors);
    fprintf('%.4f[%d]: [%s], %.4f (%.2fs) | theta\n', C, loopi, ...
        num2str(mean(theta), '%.2f '), fval, tElapsed);
    etime = etime + tElapsed;
end


end