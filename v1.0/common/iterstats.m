% collect iteration statistics
% 
% inputs:
% itercell - 'errs', 'postdim', 'fobjs', 'etimes', etc.
% rid - when 'itercell' is 'errs', use 'rid' to index the error of interest 
%       (omit it to indicate collecting all error measures)
% statfun - statistic function handle (defaults to 'min')
% 
% outputs:
% endvals - the last value when iteration terminates
% iters - number of iterations
% statvals - specific statistic
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function [endvals, iters, statvals] = iterstats(itercell, rid, statfun)
% maxiter = 0;
% for i = 1:numel(itercell)
%     if numel(itercell{i}) > maxiter
%         maxiter = numel(itercell{i});
%     end
% end
if ~exist('statfun', 'var')
    statfun = @(x)min(x, [], 2);
end
if ~exist('rid', 'var')
    rid = ':';
end
endvals = cell(size(itercell));
statvals = cell(size(itercell));
iters = zeros(size(itercell));
figure, hold on;
for i = 1:numel(itercell)
    if isempty(itercell{i})
        endvals{i} = nan;
        statvals{i} = nan;
        continue;
    end
    endvals{i} = itercell{i}(rid(:),end);
    statvals{i} = statfun(itercell{i}(rid(:),:));
    iters(i) = numel(itercell{i}(1,:));
    if numel(rid)==1 && ~ischar(rid)
        plot(itercell{i}(rid,:));
    else
        plot(itercell{i}'*diag(1./max(itercell{i},[],2)));
    end
end
hold off;

if numel(endvals{1}) == 1
    endvals = cell2mat(endvals);
end
if numel(statvals{1}) == 1
    statvals = cell2mat(statvals);
end

end