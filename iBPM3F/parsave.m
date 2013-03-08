function parsave(fname, vars_opt, errs, fobjs, etimes, postdim, K, l, alphav, C, rho, varsigma, dobuffer)
varname = {'vars_opt', 'errs', 'fobjs', 'etimes', 'postdim', 'K', 'l', 'alphav', 'C', 'rho', 'varsigma'};
if exist(fname, 'file')
    save(fname, varname{:}, '-append');
else
    save(fname, varname{:});
end

if dobuffer
    bfrd_vars_opt = vars_opt;
    bfrd_iter = size(errs, 2);
    save(fname, 'bfrd_vars_opt', 'bfrd_iter', '-append');
end

end