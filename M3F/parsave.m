function parsave(fname, optU, optV, optheta, errs, fobjs, etimes, regvals, K, l, dobuffer)
varname = {'optU', 'optV', 'optheta', 'errs', 'fobjs', 'etimes', 'regvals', 'K', 'l'};
if exist(fname, 'file')
    save(fname, varname{:}, '-append');
else
    save(fname, varname{:});
end

if dobuffer
    bfrd_vars_opt.U = optU;
    bfrd_vars_opt.V = optV;
    bfrd_vars_opt.theta = optheta;
    bfrd_iter = size(errs, 2);
    save(fname, 'bfrd_vars_opt', 'bfrd_iter', '-append');
end

end