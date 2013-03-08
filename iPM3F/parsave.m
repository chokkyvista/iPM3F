% save variables during iteration 
% (useful when different regularization constants are tuned in a parallel fashino since
% a naive 'save' function is prohibited in the 'parfor' loop)
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function parsave(fname, optgamma, optpsi, optLambda, optheta, errs, fobjs, etimes, postdim, regvals, K, l, alphav, sigmav, rho, varsigma, dobuffer)
varname = {'optgamma', 'optpsi', 'optLambda', 'optheta', 'errs', 'fobjs', 'etimes', 'postdim', 'regvals', 'K', 'l', 'alphav', 'sigmav', 'rho', 'varsigma'};
if exist(fname, 'file')
    save(fname, varname{:}, '-append');
else
    save(fname, varname{:});
end

if dobuffer
    bfrd_vars_opt.gamma = optgamma;
    bfrd_vars_opt.psi = optpsi;
    bfrd_vars_opt.Lambda = optLambda;
    bfrd_vars_opt.theta = optheta;
    bfrd_iter = size(errs, 2);
    save(fname, 'bfrd_vars_opt', 'bfrd_iter', '-append');
end

end