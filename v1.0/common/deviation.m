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
    nvars = numel(varsold);
    dev = zeros(1, nvars);
    for i = 1:nvars
        dev(i) = devfun(varsold{i}, newvars{i});
    end
end

end

function y = avgelewisereldev(x1, x2)
y = mean(abs((x2(:)-x1(:))./(x1(:)+(x1(:)==0))));
end