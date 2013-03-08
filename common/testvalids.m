% test the performance of the optimal result (under validation set) on the real test set
% 
% inputs:
% vdir - root dir where the 'opts' are stored
% testdata - test set ('weaktest' or 'strongtest')
% ee - EMAE
% vort - 
%   0: collect the performance data on validation set
%   1: test on real test set
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function [minerrs, enderrs, mfobjs, iters, subdirnames] = testvalids(vdir, testdata, ee, vort)
if ~exist('vort', 'var')
    vort = 1;
end
minerrs = {};
enderrs = {};
mfobjs = [];
iters = [];
conflict = [];

subdirs = dir(vdir);
subdirnames = {};
for i = 1:numel(subdirs)
    if any(strcmp(subdirs(i).name, {'.', '..'}))
        continue;
    end
    if subdirs(i).isdir
        opts = dir(fullfile(vdir, subdirs(i).name));
        subdirnames{end+1} = subdirs(i).name;
        fprintf('Processing dir ''%s'':', subdirs(i).name);
        for j = 1:numel(opts)
            A = sscanf(opts(j).name, 'opts_%d.mat');
            if isempty(A)
                continue;
            end
            if conflict ~= 0
                warning('Output might get overwritten!');
            end
            conflict = 0;
            fprintf(' ''%s''', opts(j).name);
            [enderrs{A,numel(subdirnames)}, minerrs{A,numel(subdirnames)}, ...
             mfobjs(A,numel(subdirnames)), iters(A,numel(subdirnames))] = ...
                testvalid(load(fullfile(vdir, subdirs(i).name, opts(j).name)), ...
                testdata{A}, ee, vort);
        end
        fprintf('\n');
    else
        A = sscanf(subdirs(i).name, 'opts_%d_%d.mat');
        if isempty(A)
            continue;
        end
        if conflict ~= 1
            warning('Output might get overwritten!');
        end
        conflict = 1;
        fprintf('Processing file ''%s''\n', subdirs(i).name);
        [enderrs{A(1),A(2)}, minerrs{A(1),A(2)}, ...
         mfobjs(A(1),A(2)), iters(A(1),A(2))] = testvalid( ...
            load(fullfile(vdir, subdirs(i).name)), testdata{A(1)}, ee, vort);
    end
end

end