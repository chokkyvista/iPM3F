% Written by Minjie Xu (chokkyvista06@gmail.com)

errs = cell(numel(traindata), numel(regvals));
etimes = cell(size(errs));
fvals = cell(size(errs));
postdim = cell(size(errs));
merr = cell(size(errs));
mfval = cell(size(errs));

if any(~ismember({'rid_s','rid_t'}, who))
    rid_s = 1;
    rid_t = numel(regvals);
end

for di = 1:numel(traindata)
    fprintf('di = %d\n', di);

    if ~exist('validid', 'var') || ~validid
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
    
    for ci = rid_s:rid_t
        C = regvals(ci);
        fprintf('C = %.4f\n', C);
        
        if ~wors
            Z = Z_init;
            V = V_init;
        else
            Z = weakopts{di,ci}.Z;
            V = weakopts{di,ci}.V;
        end
        theta = theta_init;
        invlambda = invlambda_init;
        
        if algtype ~= 2
            mzZ = zeros(N,0);
            mzV = zeros(M,0);
            mtheta = zeros(size(theta));
            minvlambda = zeros(size(invlambda));
        end
        
        cerr = cell(nargout(@errmsr)-1, 1);
        cmerr = cell(size(cerr));
        minerr = 1;
        minfval = inf;
        loopi = numel(postdim{di,ci}) + 1;
        cnvg = 0;
        earlystop = 0;
        
        while true
            [Z,V,theta,invlambda,zcidx,newK,etime,fval] = mcmc(Z, V, theta, invlambda, ...
                alphav, sigmav, T, Su, Sv, C, ell, rho, varsigma, ijn, poisstrunc, ...
                algtype, loopi, wors);
            K = size(Z, 2);
            
            [cerr{:}] = errmsr(Z*V', theta, tY, ell, ee);
            if algtype ~= 2 && loopi > burnin
                navg = loopi - burnin;
                if navg == 1
                    mnZ = Z;
                    mnV = V;
                else
                    [mnZ, mzZ] = updatem(mnZ, Z, navg, zcidx, newK, mzZ);
                    [mnV, mzV] = updatem(mnV, V, navg, zcidx, newK, mzV);
                end
                mZ = [mnZ,mzZ]; mV = [mnV,mzV];
                mtheta = mtheta + (theta-mtheta)./navg;
                minvlambda = minvlambda + (invlambda-minvlambda)./navg;
                [cmerr{:}] = errmsr(mZ*mV', mtheta, tY, ell, ee);
                cmfval = fobj(mnZ, mnV, mtheta, T, Su, C, ell, rho, varsigma, sigmav, alphav, ijn, wors);
            else
                mZ = Z; mV = V; mtheta = theta; minvlambda = invlambda;
                if algtype == 2
                    cmerr = cerr; cmfval = fval;
                else
                    cmerr(:) = {nan}; cmfval = nan;
                end
            end
            
            errs{di, ci} = [errs{di, ci}, cell2mat(cerr)];
            etimes{di, ci} = [etimes{di, ci}, etime];
            fvals{di, ci} = [fvals{di, ci}, fval];
            postdim{di, ci} = [postdim{di, ci}, K];
            merr{di, ci} = [merr{di, ci}, cell2mat(cmerr)];
            mfval{di, ci} = [mfval{di, ci}, cmfval];

            fprintf('(%d,%d,%d): K = %d, nmae = %.4f, fval = %.4f, mnmae = %.4f, mfval = %.4f\n', ...
                di, ci, loopi, K, cerr{1}, fval, cmerr{1}, cmfval);

            if cachiter
                parsave([savedir filesep sprintf('opts_%d_%d.mat', di, ci)], ...
                    mZ, mV, mtheta, minvlambda, ...
                    errs{di, ci}, etimes{di, ci}, fvals{di, ci}, postdim{di, ci}, merr{di, ci}, mfval{di, ci}, ...
                    regvals, ell, alphav, sigmav, rho, varsigma, poisstrunc, algtype, burnin, ...
                    cmerr{1} < minerr);
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
            if cnvg == 10 || loopi == maxiter || earlystop >= 10 || K > 1000
                break;
            end
            loopi = loopi + 1;
        end
        if ~cachiter
            parsave([savedir filesep sprintf('opts_%d_%d.mat', di, ci)], ...
                mZ, mV, mtheta, minvlambda, ...
                errs{di, ci}, etimes{di, ci}, fvals{di, ci}, postdim{di, ci}, merr{di, ci}, mfval{di, ci}, ...
                regvals, ell, alphav, sigmav, rho, varsigma, poisstrunc, algtype, burnin, ...
                cmerr{1} < minerr);
        end
    end
    save([savedir filesep 'opts.mat'], 'errs', 'etimes', 'fvals', 'postdim', 'merr', 'mfval', ...
        'regvals', 'ell', 'alphav', 'sigmav', 'rho', 'varsigma', 'poisstrunc', 'algtype', 'burnin');
end