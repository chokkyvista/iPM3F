% Written by Minjie Xu (chokkyvista06@gmail.com)

ROOTDIR = '..';
MEXBINDIR = [ROOTDIR filesep 'bin'];
[~,~,~] = mkdir(MEXBINDIR);
addpath([ROOTDIR filesep 'common'], MEXBINDIR, fullfile(ROOTDIR, '..', 'daSVM'));

regvals = sqrt(sqrt(10)).^[8 7.5 7 6.5 6 5.5 5 4.5 4 3.5 3];
regvals = 1./regvals(end:-1:1);
K = 100;
ell = 9;
rho = linspace(0.5*(2-L)*3*ell, 0.5*(L-2)*3*ell, L-1);
varsigma = ell*1.5;

% weakopts = [];
% validid = 0;
wors = ~isempty(weakopts);
if ~wors
    traindata = weaktrain; testdata = weaktest; validata = weakvalid;
else
    traindata = strongtrain; testdata = strongtest; validata = strongvalid;
end
[N,M] = size(traindata{1});
algtype = 1;
cachiter = false;
maxiter = 50;
burnin = 10;

U_init = randn(N,K);
V_init = randn(M,K);
theta_init = rho(ones(1,N),:);

% savedir = '.';
save([savedir filesep 'init.mat'], 'U_init','V_init','theta_init');