% save variables during iteration 
% (useful when different regularization constants are tuned in a parallel fashino since
% a naive 'save' function is prohibited in the 'parfor' loop)
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function parsave(fname, U, V, theta, invlambda, errs, etimes, fvals, merr, mfval, regvals, ell, rho, varsigma, K, algtype, burnin, dobuffer)
varname = {'U', 'V', 'theta', 'invlambda', 'errs', 'etimes', 'fvals', 'merr', 'mfval', 'regvals', 'ell', 'rho', 'varsigma', 'K', 'algtype', 'burnin'};
if exist(fname, 'file')
    save(fname, varname{:}, '-append');
else
    save(fname, varname{:});
end

if dobuffer
    bfrd_vars_opt.U = U;
    bfrd_vars_opt.V = V;
    bfrd_vars_opt.theta = theta;
    bfrd_vars_opt.invlambda = invlambda;
    bfrd_iter = size(errs, 2);
    save(fname, 'bfrd_vars_opt', 'bfrd_iter', '-append');
end

end