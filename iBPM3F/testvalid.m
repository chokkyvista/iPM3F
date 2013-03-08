function [enderr, minerr, mfobj, iter] = testvalid(opt, tY, ee, vort)
mfobj = min(opt.fobjs);
iter = numel(opt.fobjs);
if vort
    enderr = cell(nargout(@errmsr)-1, 1);
    [enderr{:}] = errmsr(opt.vars_opt.psi*opt.vars_opt.Lambda', opt.vars_opt.theta, tY, opt.l, ee);
    enderr = cell2mat(enderr);
else
    enderr = opt.errs(:,end);
end
if vort
    if isfield(opt, 'bfrd_vars_opt')
        minerr = cell(size(enderr));
        [minerr{:}] = errmsr(opt.bfrd_vars_opt.psi*opt.bfrd_vars_opt.Lambda', opt.bfrd_vars_opt.theta, tY, opt.l, ee);
        minerr = cell2mat(minerr);
    else
        minerr = nan;
    end
else
    minerr = min(opt.errs, [], 2);
end
end