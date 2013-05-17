% Written by Minjie Xu (chokkyvista06@gmail.com)

ROOTDIR = '..';
MEXBINDIR = [ROOTDIR filesep 'bin'];
[~,~,~] = mkdir(MEXBINDIR);
addpath([ROOTDIR filesep 'common'], MEXBINDIR, [ROOTDIR filesep 'IBP'], fullfile(ROOTDIR, '..', 'daSVM'));

eval(['mex -O -outdir ''' MEXBINDIR ''' -output zrow_sampler CXXFLAGS="\$CXXFLAGS -std=c++0x" ''' fullfile(ROOTDIR, 'common', 'zrow_sampler.cpp') '''']);

regvals = sqrt(sqrt(10)).^[8 7.5 7 6.5 6 5.5 5 4.5 4 3.5 3];
regvals = 1./regvals(end:-1:1);
ell = 9;
alphav = 3;
sigmav = 1;
rho = linspace(0.5*(2-L)*3*ell, 0.5*(L-2)*3*ell, L-1);
varsigma = ell*1.5;

wors = exist('weakopts', 'var') && ~isempty(weakopts);
if ~wors
    traindata = weaktrain; testdata = weaktest; validata = weakvalid;
else
    traindata = strongtrain; testdata = strongtest; validata = strongvalid;
end
[N,M] = size(traindata{1});
poisstrunc = max(ceil(alphav/N+5*sqrt(alphav/N)), 10);
algtype = 1;
cachiter = false;
maxiter = 50;
burnin = 10;

Z_init = ibprnd(N, alphav);
K = size(Z_init, 2);
V_init = sigmav*randn(M,K);
theta_init = rho(ones(1,N),:);

savedir = input('save to: ', 's');
mkdir(savedir);
save([savedir filesep 'init.mat'], 'Z_init','K','V_init','theta_init');