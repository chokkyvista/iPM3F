% calculate deviation between 'varsold' & 'newvars' to help determine
% the convergence of variational inference
%
% 'varsold' & 'newvars' can be
% 1. structs with same fields
% 2. cells with same number of elements
% and the returned 'dev' is a vector containing the corresponding deviations
%
% 'devfun': defaults to AVErage-ELEment-WISE-RELative-DEVation
%
% Written by Minjie Xu (chokkyvista06@gmail.com)

function [dev] = deviation(varsold, newvars, devfun)
if ~exist('devfun','var')
    devfun = @avgelewisereldev;
end
if isstruct(varsold)
    vars = fieldnames(varsold);
    nvars = numel(vars);
    dev = zeros(1, nvars);
    for i = 1:nvars
        dev(i) = devfun(getfield(varsold,vars{i}), getfield(newvars,vars{i}));
    end
elseif iscell(varsold)
    dev = cellfun(devfun, varsold, newvars);
end

end

function y = avgelewisereldev(x1, x2)
y = mean(abs((x2(:)-x1(:))./(x1(:)+(x1(:)==0))));
end