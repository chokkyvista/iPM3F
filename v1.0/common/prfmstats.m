% Display performance statistics
% 
% opts - the structure 'opts' as output by, e.g., 'train_ipm3f'
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function prfmstats(opts, di, ci)
fprintf('regvals:\n');
fprintf('    %.4f', opts.regvals(ci));
fprintf('\niters:\n');
[endvals, iters, minvals] = iterstats(opts.errs(di,ci), 1);
fprintf('      %2d', round(mean(iters(:,1), 1)));
fprintf('        %2d', round(mean(iters(:,2:end), 1)));
fprintf('\nmean(nmae):\n');
fprintf('    %.4f', mean(endvals, 1));
fprintf('\n');
fprintf('    %.4f', mean(minvals, 1));
pause
[endvals, ~, minerrs] = iterstats(opts.merr(di,ci), 1);
fprintf('\nmean(mnmae):\n');
fprintf('    %.4f', mean(endvals, 1));
fprintf('\n');
fprintf('    %.4f', mean(minerrs, 1));
pause
[endvals, ~, minvals] = iterstats(opts.fvals(di,ci), 1);
fprintf('\nmean(fval): 1e6\n');
fprintf('    %.4f', mean(endvals./1e6, 1));
fprintf('\n');
fprintf('    %.4f', mean(minvals./1e6, 1));
pause
[endvals, ~, minvals] = iterstats(opts.mfval(di,ci), 1);
fprintf('\nmean(mfval): 1e6\n');
fprintf('    %.4f', mean(endvals./1e6, 1));
fprintf('\n');
fprintf('    %.4f', mean(minvals./1e6, 1));
pause
if isfield(opts, 'postdim')
    [endvals, ~, inivals] = iterstats(opts.postdim(di,ci), 1, @(x)x(1));
    fprintf('\nmean(postdim):\n');
    fprintf('     %4d', round(mean(endvals(:,1), 1)));
    fprintf('      %4d', round(mean(endvals(:,2:end), 1)));
    fprintf('\n');
    fprintf('     %4d', round(mean(inivals(:,1), 1)));
    fprintf('      %4d', round(mean(inivals(:,2:end), 1)));
    fprintf('\n');
    pause
end
fprintf('\n');

[~, optcid] = min(mean(minerrs));
miniter = min(iters(:,optcid));
optcid = ci(optcid);
figure;
plot(cell2mat(cellfun(@(x)x(1:miniter), opts.fvals(di,optcid), 'UniformOutput', false))', ...
    '*--', 'LineWidth', 1.5);
hold on;
plot(cell2mat(cellfun(@(x)x(1:miniter), opts.mfval(di,optcid), 'UniformOutput', false))', ...
    '.-', 'LineWidth', 1.5);
pause
figure;
plot(cell2mat(cellfun(@(x)x(1,1:miniter), opts.errs(di,optcid), 'UniformOutput', false))', ...
    '*--', 'LineWidth', 1.5);
hold on;
plot(cell2mat(cellfun(@(x)x(1,1:miniter), opts.merr(di,optcid), 'UniformOutput', false))', ...
    '.-', 'LineWidth', 1.5);
pause

close all

end