% visualize the IBP samples (whether binary or real-valued)
% 
% Z - the IBP sample
% blocksz - side length of the square block to represent each entry
% clrng - color range (defaults to [0.2,0.8] so that the minimal(maximal) non-zero entry
%         will be light(dark) gray)
% Zonly - if false, extra info will be inserted (e.g. colormap)
% 
% See also IBPRND, RIBPRND
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function im = visualibp(Z, blocksz, clrng, Zonly)
if ~exist('blocksz', 'var')
    blocksz = 20;
end
if ~exist('clrng', 'var')
    clrng = [0.2,0.8];
end
if ~exist('Zonly', 'var')
    Zonly = true;
end

blockmgn = 1;
vgap = max(2,ceil(blocksz/4));

nzvals = nonzeros(Z);
maxzval = max(nzvals);
minzval = min(nzvals);
ca = diff(clrng)./(minzval-maxzval);
function r = mapclr(v)
    r(v==0) = 1;
    if maxzval ~= minzval
        r(v~=0) = ca*(v-maxzval)+clrng(1);
    else
        r(v~=0) = 0;
    end
end

[N,K] = size(Z);
im = zeros(blocksz*N+2*blockmgn*N+vgap*(N-1), blocksz*K+blockmgn*(K+1));
cbvi = blockmgn+1;
for i = 1:N
    cbhj = blockmgn+1;
    for j = 1:K
        im(cbvi:cbvi+blocksz-1, cbhj:cbhj+blocksz-1) = mapclr(Z(i,j));
        cbhj = cbhj+blockmgn+blocksz;
    end
    cbvi = cbvi+2*blockmgn+vgap+blocksz;
    if (cbvi < size(im,1))
        im(cbvi-blockmgn-vgap:cbvi-blockmgn-1,:) = 1;
    end
end

imshow(im);
if ~Zonly
    colorbar('YLim', clrng, 'YDir', 'reverse', 'YMinorTick', 'on', 'YTick', linspace(clrng(1),clrng(2),11), 'YTickLabel', cellstr(num2str(linspace(1,0,11)')));
end

end