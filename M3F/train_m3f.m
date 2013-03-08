errs = cell(numel(traindata), numel(regvals));
fobjs = cell(size(errs));
etimes = cell(size(errs));
maxiter = 50;

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
    
    for ci = rid_s:rid_t %1:numel(regvals)
        C = regvals(ci);
        fprintf('C = %.4f\n', C);
%         fprintf('h = %.4f\n', C);
        
        U = U_init;
        if ~wors
            V = V_init;
        else
            V = weakopts{di, ci};
        end
        theta = theta_init;
        fobj = fObj(U, V, theta, Su, L, N, C, l, T_rjs, wors);

        cerr = cell(nargout(@errmsr)-1, 1);
        minerr = 1;
        loopi = 1;
        cnvg = 0;
        earlystop = 0;
        while true
            U_old = U;
            V_old = V;
            theta_old = theta;
            fobj_old = fobj;

            [U,V,theta,fobj,etime] = bilinsvm(U, V, theta, Su, Sv, L, N, M, K, C, l, T_ris, T_rjs, loopi, wors);
%             [U,V,theta] = bilinsvm(U, V, theta, Su, Sv, L, N, M, K, 0.045, C, T_ris, T_rjs, loopi, wors);
            
            delta_U = norm(U_old(:)-U(:))/sqrt(numel(U));
            delta_V = norm(V_old(:)-V(:))/sqrt(numel(V));
            delta_theta = norm(theta_old(:)-theta(:))/sqrt(numel(theta));
            [cerr{:}] = errmsr(U*V', theta, tY, l, ee);
            errs{di, ci} = [errs{di, ci}, cell2mat(cerr)];
            fobjs{di, ci} = [fobjs{di, ci}, fobj];
            etimes{di, ci} = [etimes{di, ci}, etime];
            delta_fobj = (fobj_old - fobj) / fobj_old;
            fprintf('(%d,%d,%d): dU = %.3f, dV = %.3f, dtheta = %.3f, df = %.3f\n', ...
                di, ci, loopi, delta_U, delta_V, delta_theta, delta_fobj);

            parsave([savedir filesep sprintf('opts_%d_%d.mat', di, ci)], ...
                U, V, theta, errs{di, ci}, fobjs{di, ci}, etimes{di, ci}, ...
                regvals, K, l, ...
                cerr{1} < minerr);
            minerr = min(cerr{1}, minerr);

            if all([delta_U, delta_V, delta_theta] < 0.01) || delta_fobj < 0.001
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
    save([savedir filesep 'opts.mat'], 'errs', 'fobjs', 'etimes', 'regvals', 'K', 'l');
%     save([savedir filesep 'opts.mat'], 'errs', 'fobjs', 'etimes', 'regvals', 'K');
end
