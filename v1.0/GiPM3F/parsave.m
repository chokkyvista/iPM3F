% save variables during iteration 
% (useful when different regularization constants are tuned in a parallel fashino since
% a naive 'save' function is prohibited in the 'parfor' loop)
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function parsave(fname, Z, V, theta, invlambda, errs, etimes, fvals, postdim, merr, mfval, regvals, ell, alphav, sigmav, rho, varsigma, poisstrunc, algtype, burnin, dobuffer)
varname = {'Z', 'V', 'theta', 'invlambda', 'errs', 'etimes', 'fvals', 'postdim', 'merr', 'mfval', 'regvals', 'ell', 'alphav', 'sigmav', 'rho', 'varsigma', 'poisstrunc', 'algtype', 'burnin'};
if exist(fname, 'file')
    save(fname, varname{:}, '-append');
else
    save(fname, varname{:});
end

if dobuffer
    bfrd_vars_opt.Z = Z;
    bfrd_vars_opt.V = V;
    bfrd_vars_opt.theta = theta;
    bfrd_vars_opt.invlambda = invlambda;
    bfrd_iter = size(errs, 2);
    save(fname, 'bfrd_vars_opt', 'bfrd_iter', '-append');
end

end