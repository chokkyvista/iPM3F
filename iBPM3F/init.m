%% 
ROOTDIR = '..';
MEXBINDIR = [ROOTDIR filesep 'bin'];
[~,~,~] = mkdir(MEXBINDIR);
addpath([ROOTDIR filesep 'common'], MEXBINDIR);

%%
mexcmd = ['mex -O -outdir ''' MEXBINDIR ''' -output bilinear_svm_opt -DMEX -I''' ROOTDIR filesep 'SVM_Multiclass'' -I''' ROOTDIR filesep 'SVMLight'''];

% Append all source file names to the MEX command string
SRCCPP = { 
    ['common' filesep 'bilinear_svm_opt.cpp']
    ['SVMLight' filesep 'svm_common.cpp']
    ['SVMLight' filesep 'svm_hideo.cpp']
    ['SVMLight' filesep 'svm_learn.cpp']
    ['SVM_Multiclass' filesep 'svm_struct_api.cpp']
    ['SVM_Multiclass' filesep 'svm_struct_common.cpp']
    ['SVM_Multiclass' filesep 'svm_struct_learn.cpp']
};
for f = 1:length(SRCCPP)
    mexcmd = [mexcmd ' ''' ROOTDIR filesep SRCCPP{f} ''' '];
end

eval(mexcmd);  % compile and link in one step

eval(['mex -O -outdir ''' MEXBINDIR ''' -output gs_search_psivi ''' fullfile(ROOTDIR, 'common', 'gs_search_psivi.cpp') '''']);
eval(['mex -O -outdir ''' MEXBINDIR ''' -output bcd_search_shpcr ''' fullfile(ROOTDIR, 'common', 'bcd_search_shpcr.cpp') '''']);

%% 
regvals = sqrt(sqrt(10)).^[8 7.5 7 6.5 6 5.5 5 4.5 4 3.5 3];
regvals = 1./regvals(end:-1:1);
K = 100;
l = 9;
alphav = 3;
C = 1;
rho = linspace((2-L)*l, (L-2)*l, L-1);
varsigma = l*1.5;

wors = exist('weakopts', 'var') && ~isempty(weakopts);
if ~wors
    traindata = weaktrain; testdata = weaktest; validata = weakvalid;
else
    traindata = strongtrain; testdata = strongtest; validata = strongvalid;
end
[N,M] = size(traindata{1});


%%
vars_init.gamma = repmat([alphav, 1], K, 1);
% vars_init.psi = repmat(cumprod(betarnd(alphav, 1, 1, K)), N, 1);
vars_init.psi = rand(N, K);
vars_init.Lambda = randn(M, K);
vars_init.theta = repmat(rho, N, 1);
vars_init.mu = zeros(1,K);
vars_init.beta = 1;
vars_init.invW = eye(K);
vars_init.tau = K;
vars_init.invXi = vars_init.invW./vars_init.tau;

savedir = input('save to: ', 's');
mkdir(savedir);
save([savedir filesep 'init.mat'], 'vars_init');