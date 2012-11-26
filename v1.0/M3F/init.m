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

eval(['mex -O -outdir ''' MEXBINDIR ''' -output bcd_search_shpcr ''' fullfile(ROOTDIR, 'common', 'bcd_search_shpcr.cpp') '''']);

%% 
regvals = sqrt(sqrt(10)).^[8 7.5 7 6.5 6 5.5 5 4.5 4 3.5 3];
regvals = 1./regvals(end:-1:1);
K = 100;
l = 9;

wors = ~isempty(weakopts);
if ~wors
    traindata = weaktrain; testdata = weaktest; validata = weakvalid;
else
    traindata = strongtrain; testdata = strongtest; validata = strongvalid;
end
[N,M] = size(traindata{1});

%%
U_init = randn(N, K);
V_init = randn(M, K);
theta_init = repmat(linspace((2-L)*l, (L-2)*l, L-1), N, 1);

save([savedir filesep 'init.mat'], 'U_init', 'V_init', 'theta_init');