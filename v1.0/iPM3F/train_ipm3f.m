errs = cell(numel(traindata), numel(regvals));
fobjs = cell(size(errs));
etimes = cell(size(errs));
postdim = cell(size(errs));
maxiter = 50;

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
    T_rjs = cell(1, N);
    for i = 1:N
        Su{i} = find(Y(i,:));
        T_rjs{i} = repmat(Y(i,Su{i}), L-1, 1) <= repmat((1:L-1)', 1, numel(Su{i}));
    end
    Sv = cell(1, M);
    T_ris = cell(1, M);
    for j = 1:M
        Sv{j} = find(Y(:,j));
        T_ris{j} = repmat(Y(Sv{j},j)', L-1, 1) <= repmat((1:L-1)', 1, numel(Sv{j}));
    end
    
%     gamma_inits = cell(1, numel(regvals));
%     psi_inits = cell(size(gamma_inits));
%     Lambda_inits = cell(size(gamma_inits));
%     theta_inits = cell(size(gamma_inits));
%     fobjs_inits = cell(size(gamma_inits));
%     errs_inits = cell(size(gamma_inits));
%     etimes_inits = cell(size(gamma_inits));
%     postdim_inits = cell(size(gamma_inits));
%     for ci = rid_s:rid_t
%         A = load(num2str([di, ci], [savedir filesep 'opts_%d_%d.mat']));
%         gamma_inits{ci} = A.optgamma;
%         psi_inits{ci} = A.optpsi;
%         Lambda_inits{ci} = A.optLambda;
%         theta_inits{ci} = A.optheta;
%         fobjs_inits{ci} = A.fobjs;
%         errs_inits{ci} = A.errs;
%         etimes_inits{ci} = A.etimes;
%         postdim_inits{ci} = A.postdim;
%     end

    for ci = rid_s:rid_t %1:numel(regvals)
        C = regvals(ci);
        fprintf('C = %.4f\n', C);
        
        gammav = gamma_init;
        U = psi_init;
        if ~wors
            V = Lambda_init;
        else
            V = weakopts{di,ci};
        end
        theta = theta_init;
%         gammav = gamma_inits{ci};
%         U = psi_inits{ci};
%         V = Lambda_inits{ci};
%         theta = theta_inits{ci};
%         fobjs{di, ci} = fobjs_inits{ci};
%         errs{di, ci} = errs_inits{ci};
%         etimes{di, ci} = etimes_inits{ci};
%         postdim{di, ci} = postdim_inits{ci};
        fobj = fObj(gammav, U, V, theta, alphav, sigmav, Su, L, N, K, C, l, T_rjs, wors);
        
        cerr = cell(nargout(@errmsr)-1, 1);
        minerr = 1;
        loopi = numel(fobjs{di,ci}) + 1;
        cnvg = 0;
        earlystop = 0;
        while true
            gamma_old = gammav;
            U_old = U;
            V_old = V;
            theta_old = theta;
            fobj_old = fobj;

            [gammav,U,V,theta,fobj,etime] = varinfr(gammav, U, V, theta, ...
                alphav, sigmav, Su, Sv, L, N, M, K, C, l, rho, varsigma, ...
                T_ris, T_rjs, loopi, wors);
            
            delta_gamma = norm(gamma_old(:)-gammav(:))/sqrt(numel(gammav));
            delta_U = norm(U_old(:)-U(:))/sqrt(numel(U));
            delta_V = norm(V_old(:)-V(:))/sqrt(numel(V));
            delta_theta = norm(theta_old(:)-theta(:))/sqrt(numel(theta));
            [cerr{:}] = errmsr(U*V', theta, tY, l, ee);
            errs{di, ci} = [errs{di, ci}, cell2mat(cerr)];
            fobjs{di, ci} = [fobjs{di, ci}, fobj];
            etimes{di, ci} = [etimes{di, ci}, etime];
            postdim{di, ci} = [postdim{di, ci}, K-sum(prod(1-U))];
            delta_fobj = (fobj_old - fobj) / fobj_old;
            fprintf('(%d,%d,%d): dgamma = %.3f, dpsi = %.3f, dLambda = %.3f, dtheta = %.3f, df = %.3f\n', ...
                di, ci, loopi, delta_gamma, delta_U, delta_V, delta_theta, delta_fobj);

            parsave([savedir filesep sprintf('opts_%d_%d.mat', di, ci)], ...
                gammav, U, V, theta, ...
                errs{di, ci}, fobjs{di, ci}, etimes{di, ci}, postdim{di, ci}, ...
                regvals, K, l, alphav, sigmav, rho, varsigma, ...
                cerr{1} < minerr);
            minerr = min(cerr{1}, minerr);
            
            if all([delta_gamma, delta_U, delta_V, delta_theta] < 0.01) ...
                || delta_fobj < 0.001%|| abs(delta_fobj) < 0.005
                cnvg = cnvg + 1;
            else
                cnvg = 0;
            end
            if loopi > 4 && cerr{1} > errs{di, ci}(1,end-4)
                earlystop = earlystop + 2;
            elseif cerr{1} > minerr
                earlystop = earlystop + 1;
            else
                earlystop = 0;
            end
            if cnvg == 10 || loopi == maxiter || earlystop >= 10
                break;
            end
            loopi = loopi + 1;
        end
    end
    save([savedir filesep 'opts.mat'], 'errs', 'fobjs', 'etimes', 'postdim', ...
        'regvals', 'K', 'l', 'alphav', 'sigmav', 'rho', 'varsigma');
end
