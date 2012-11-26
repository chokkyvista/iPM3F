% partition data set
% 
% requires 'sparse sub access' by Bruno Luong:
% http://www.mathworks.com/matlabcentral/fileexchange/23488-sparse-sub-access
% 
% inputs:
% ratings - the entire rating matrix (N*M, 0 for missing value, ratings in 1,2,...,L)
% minrpu - minimum number of ratings per user (filter the incapable users)
% minrpm - minimum number of ratings per movie (filter the incapable movies)
% nsets - perform random partition for 'nsets' times
% nweakusrs - number of 'weak' users (omit it to indicate using all users for 'weak')
% 
% WARNING: when 'minrpu' and 'minrpm' are both non-zero, the resulting matrix may not
% necessarily meet the requirements
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function [weaktrain, weaktest, strongtrain, strongtest] = pardataset(ratings, minrpu, minrpm, nsets, nweakusrs)
weaktrain = cell(1, nsets);
weaktest = cell(size(weaktrain));
strongtrain = cell(size(weaktrain));
strongtest = cell(size(strongtrain));

if issparse(ratings)
    ratings = full(ratings);
end
obsvrids = ratings>=1;
mids = find(sum(obsvrids, 1) >= minrpm);
uids = find(sum(obsvrids, 2) >= minrpu);
if ~exist('nweakusrs', 'var')
    nweakusrs = numel(uids);
elseif numel(uids) < nweakusrs
    warning('Not enough users are left for the training set after pruning!');
end
nstrongusrs = numel(uids) - nweakusrs;

usrmids = cell(1, numel(uids));
for i = 1:numel(uids)
    usrmids{i} = find(ratings(uids(i),mids) >= 1);
end
tstmids = zeros(size(usrmids));

for i = 1:nsets
    if nweakusrs == numel(uids)
        wuinds = 1:nweakusrs;
    else
        rpuinds = randperm(numel(uids));
        wuinds = rpuinds(1:nweakusrs);
    end
    for j = 1:numel(uids)
        tstmids(j) = usrmids{j}(randi(numel(usrmids{j})));
    end
    
%     [tmids, tuids] = meshgrid(mids, uids(wuinds));
%     weaktrain{i} = reshape(getsparse(ratings, tuids(:), tmids(:)), nweakusrs, numel(mids));
    weaktrain{i} = sparse(ratings(uids(wuinds), mids));
    weaktest{i} = setsparse(sparse(nweakusrs, numel(mids)), 1:nweakusrs, tstmids(wuinds), getsparse(weaktrain{i}, 1:nweakusrs, tstmids(wuinds)));
%     weaktest{i} = sparse(1:nweakusrs, tstmids(wuinds), getsparse(weaktrain{i}, 1:nweakusrs, tstmids(wuinds)), nweakusrs, numel(mids));
    weaktrain{i} = weaktrain{i} - weaktest{i};
    
    if nweakusrs == numel(uids)
        continue;
    end
    suinds = rpuinds(nweakusrs+1:end);
%     [tmids, tuids] = meshgrid(mids, uids(suinds));
%     strongtrain{i} = reshape(getsparse(ratings, tuids(:), tmids(:)), nstrongusrs, numel(mids));
    strongtrain{i} = sparse(ratings(uids(suinds), mids));
    strongtest{i} = setsparse(sparse(nstrongusrs, numel(mids)), 1:nstrongusrs, tstmids(suinds), getsparse(strongtrain{i}, 1:nstrongusrs, tstmids(suinds)));
%     strongtest{i} = sparse(1:nstrongusrs, tstmids(suinds), getsparse(strongtrain{i}, 1:nstrongusrs, tstmids(suinds)), nstrongusrs, numel(mids));
    strongtrain{i} = strongtrain{i} - strongtest{i};
end

end