% generate 'ratings' matrix from the raw EachMovie .dat file
% 
% requires 'sparse sub access' by Bruno Luong:
% http://www.mathworks.com/matlabcentral/fileexchange/23488-sparse-sub-access
%
% Written by Minjie Xu (chokkyvista06@gmail.com)

function [ratings, rawdata] = eachmovie(datfile, sp)
if ~exist('sp', 'var')
    sp = true;
end
rawdata = fscanf(datfile, '%d\t%d\t%f\t%f\t%*s %*s', [4, inf]);

rids = (rawdata(1,:)-1)*max(rawdata(2,:))+rawdata(2,:);
dblratings = find(~(rids(1:end-1)-rids(2:end)));
fprintf('Detected %d double ratings:\n', numel(dblratings));
[rawdata(1,dblratings);rawdata(2,dblratings)]

if sp
    % sparse matrix
    % ratings = sparse(rawdata(1,:), rawdata(2,:), 0.2+rawdata(3,:)-0.4*(rawdata(4,:)~=1).*(~rawdata(3,:))); % buggy
    ratings = spalloc(74424, 1648, size(rawdata,2));
    ratings = setsparse(ratings, rawdata(1,:), rawdata(2,:), 0.2+rawdata(3,:)-0.4*(rawdata(4,:)~=1).*(~rawdata(3,:)));
else
    % full matrix
    ratings = zeros(74424, 1648);
    ratings(sub2ind(size(ratings), rawdata(1,:), rawdata(2,:))) = 0.2+rawdata(3,:)-0.4*(rawdata(4,:)~=1).*(~rawdata(3,:));
end

ratings = round(ratings*5); % map to {-1,0,1,2,3,4,5,6}
ratings = ratings(any(ratings,2),any(ratings,1));

for r = [-1,1:6]
    fprintf('#%d: %d\n', r, numel(find(ratings==r)));
end
fprintf('#valid ratings: %d\n', nnz(ratings));

end