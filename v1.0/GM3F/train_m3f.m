% Written by Minjie Xu (chokkyvista06@gmail.com)

errs = cell(numel(traindata), numel(regvals));
etimes = cell(size(errs));
fvals = cell(size(errs));
merr = cell(size(errs));
mfval = cell(size(errs));

% rid_s = 4;
% rid_t = 4;

for di = 1:numel(traindata)
    fprintf('di = %d\n', di);

    if ~validid
        Y = full(traindata{di});
        tY = testdata{di};
    else
        Y = full(traindata{di}-validata{di}{validid});
        tY = validata{di}{validid};
    end
    Su = cell(1, N);
    for i = 1:N
        Su{i} = find(Y(i,:));
    end
    Sv = cell(1, M);
    for j = 1:M
        Sv{j} = find(Y(:,j));
    end
    ijn = zeros(N, M);
    n = 1;
    for j = 1:M
        ijn(Sv{j},j) = n:n+numel(Sv{j})-1;
        n = n + numel(Sv{j});
    end
    T = 2*(repmat(nonzeros(Y), 1, L-1)<=repmat(1:L-1,nnz(Y),1))-1;
    invlambda_init = rand(size(T));
    
    for ci = rid_s:rid_t %1:numel(regvals)
        C = regvals(ci);
        fprintf('C = %.4f\n', C);
        
        U = U_init;
        if ~wors
            V = V_init;
        else
            V = weakopts{di,ci};
        end
        theta = theta_init;
        invlambda = invlambda_init;

        mU = zeros(size(U));
        mV = zeros(size(V));
        mtheta = zeros(size(theta));
        minvlambda = zeros(size(invlambda));
        
        cerr = cell(nargout(@errmsr)-1, 1);
        cmerr = cell(size(cerr));
        minerr = 1;
        minfval = inf;
        loopi = numel(fvals{di,ci}) + 1;
        cnvg = 0;
        earlystop = 0;
        
        while true
            [U,V,theta,invlambda,etime,fval] = mcmc(U, V, theta, invlambda, ...
                T, Su, Sv, C, ell, rho, varsigma, ijn, algtype, loopi, wors);
            
            [cerr{:}] = errmsr(U*V', theta, tY, ell, ee);
            if algtype ~= 2 && loopi > burnin
                mU = mU + (U-mU)./(loopi-burnin);
                mV = mV + (V-mV)./(loopi-burnin);
                mtheta = mtheta + (theta-mtheta)./(loopi-burnin);
                minvlambda = minvlambda + (invlambda-minvlambda)./(loopi-burnin);
                [cmerr{:}] = errmsr(mU*mV', mtheta, tY, ell, ee);
                cmfval = fobj(mU, mV, mtheta, T, Su, C, ell, rho, varsigma, ijn, wors);
            else
                mU = U; mV = V; mtheta = theta; minvlambda = invlambda;
                cmerr(:) = {nan}; cmfval = nan;
            end
            
            errs{di, ci} = [errs{di, ci}, cell2mat(cerr)];
            etimes{di, ci} = [etimes{di, ci}, etime];
            fvals{di, ci} = [fvals{di, ci}, fval];
            merr{di, ci} = [merr{di, ci}, cell2mat(cmerr)];
            mfval{di, ci} = [mfval{di, ci}, cmfval];
            
            fprintf('(%d,%d,%d): nmae = %.4f, fval = %.4f, mnmae = %.4f, mfval = %.4f\n', ...
                di, ci, loopi, cerr{1}, fval, cmerr{1}, cmfval);
            
            if cachiter
                parsave([savedir filesep sprintf('opts_%d_%d.mat', di, ci)], ...
                    mU, mV, mtheta, minvlambda, ...
                    errs{di, ci}, etimes{di, ci}, fvals{di, ci}, merr{di, ci}, mfval{di, ci}, ...
                    regvals, ell, rho, varsigma, K, algtype, burnin, cmerr{1} < minerr);
            end
            minerr = min(cmerr{1}, minerr);
            
            if algtype == 2
                if fval > minfval
                    cnvg = cnvg + 2;
                else
                    if all(deviation({minfval}, {fval}) < 1e-5)
                        cnvg = cnvg + 1;
                    else
                        cnvg = 0;
                    end
                    minfval = fval;
                end
            else
                if all(deviation({minfval}, {cmfval}) < 1e-5)
                    cnvg = cnvg + 1;
                else
                    cnvg = 0;
                end
                minfval = min(minfval, cmfval);
            end
            if loopi > 4 && cmerr{1} > merr{di, ci}(1,end-4)
                earlystop = earlystop + 2;
            elseif cmerr{1} > minerr
                earlystop = earlystop + 1;
            else
                earlystop = 0;
            end
            if cnvg == 10 || loopi == maxiter || earlystop >= 10
                break;
            end
            loopi = loopi + 1;
        end
        if ~cachiter
            parsave([savedir filesep sprintf('opts_%d_%d.mat', di, ci)], ...
                mU, mV, mtheta, minvlambda, ...
                errs{di, ci}, etimes{di, ci}, fvals{di, ci}, merr{di, ci}, mfval{di, ci}, ...
                regvals, ell, rho, varsigma, K, algtype, burnin, false);
        end
    end
    save([savedir filesep 'opts.mat'], 'errs', 'etimes', 'fvals', 'merr', 'mfval', ...
        'regvals', 'ell', 'rho', 'varsigma', 'K', 'algtype', 'burnin');
end