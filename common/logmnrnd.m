% sample from Multinomial distribution with log-probabilities
% 
% This sampler directly works on the log-probabilities and hence is more
% numerically stable when exp(logp) might overflow.
% 
% inputs:
% logp - logarithm of multinomial probabilities: [log(p1), ..., log(pk)]
% 
% output: 
% r - returned sample
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function r = logmnrnd(logp)
[lsp, lcsp] = logsump(logp);
r = find(log(rand)+lsp > [-inf,lcsp], 1, 'last');

end

% lcsp = log(cumsum(exp(logp))) and again is calculated in a more
% numerically stable fashion
function [lsp, lcsp] = logsump(logp)
pivot = max(logp);
lcsp = pivot + log(cumsum(exp(logp(:)'-pivot)));
lsp = lcsp(end);

end