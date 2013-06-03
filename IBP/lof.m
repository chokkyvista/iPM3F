% take the 'lof' operation to Z
% 
% Z - can be either a binary or a real-valued matrix (compatible)
%     (we adopt the lexigraphical order, also known as the dictionary order, in both cases)
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function Z = lof(Z)
Z = sortrows(Z', -1:-1:-size(Z,1))';

end
