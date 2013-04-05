errs = cell(1, numel(traindata));
fobjs = cell(size(errs));
etimes = cell(size(errs));
postdim = cell(size(errs));
maxiter = 50;
varstr = char(strcat(' d', fieldnames(vars_init)', ' = %.3f,'))';
varstr = varstr(:)';

for di = 1:numel(traindata)
    fprintf('di = %d\n', di);
    fprintf('C = %.4f\n', C);
    
    if ~exist('validid', 'var') || ~validid
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
    
%     A = load([savedir filesep num2str(di, 'opts_%d.mat')]);
%     vars_init = A.vars_opt;
%     errs{di} = A.errs;
%     fobjs{di} = A.fobjs;
%     etimes{di} = A.etimes;
%     postdim{di} = A.postdim;
    vars_opt = vars_init;
    if wors
        vars_opt.Lambda = weakopts{di};
    end

%     vars_init.beta = 1;
%     vars_init.invW = eye(K);
%     vars_init.tau = K;
%     vars_init.invXi = vars_init.invW./vars_init.tau;
    fobj = fObj(vars_opt, alphav, Su, L, M, N, K, l, T_rjs, C, vars_init, wors);
    
    cerr = cell(nargout(@errmsr)-1, 1);
    minerr = 1;
    loopi = numel(fobjs{di}) + 1;
    cnvg = 0;
    earlystop = 0;
    while true
        vars_old = vars_opt;
        fobj_old = fobj;
        
        [vars_opt, fobj, etime] = varinfr(vars_opt, vars_init, alphav, ...
            Su, Sv, L, N, M, K, l, rho, varsigma, T_ris, T_rjs, C, loopi, wors);

        delta_vars = deviation(vars_old, vars_opt);
        [cerr{:}] = errmsr(vars_opt.psi*vars_opt.Lambda', vars_opt.theta, tY, l, ee);
        errs{di} = [errs{di}, cell2mat(cerr)];
        fobjs{di} = [fobjs{di}, fobj];
        etimes{di} = [etimes{di}, etime];
        postdim{di} = [postdim{di}, K-sum(prod(1-vars_opt.psi))];
        delta_fobj = (fobj_old - fobj) / fobj_old;
        delta_varscell = num2cell(delta_vars);
        fprintf(['(%d,%d): ' varstr ' df = %.3f\n'], di, loopi, ...
            delta_varscell{:}, delta_fobj);

        parsave([savedir filesep sprintf('opts_%d.mat', di)], ...
            vars_opt, errs{di}, fobjs{di}, etimes{di}, postdim{di}, ...
            K, l, alphav, C, rho, varsigma, ...
            cerr{1} < minerr);
        minerr = min(cerr{1}, minerr);

        if all(delta_vars<0.01) || abs(delta_fobj) < 0.001
            cnvg = cnvg + 1;
        else
            cnvg = 0;
        end
        if loopi > 4 && cerr{1} > errs{di}(1,end-4)
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
    save([savedir filesep 'opts.mat'], 'errs', 'fobjs', 'etimes', 'postdim', ...
        'K', 'l', 'alphav', 'C');
end
