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
if opts.algtype ~= 2 % not EM
    [endvals, ~, minerrs] = iterstats(opts.merr(di,ci), 1);
    fprintf('\nmean(mnmae):\n');
    fprintf('    %.4f', mean(endvals, 1));
    fprintf('\n');
    fprintf('    %.4f', mean(minerrs, 1));
    pause
end
[endvals, ~, minvals] = iterstats(opts.fvals(di,ci), 1);
fprintf('\nmean(fval): 1e6\n');
fprintf('    %.4f', mean(endvals./1e6, 1));
fprintf('\n');
fprintf('    %.4f', mean(minvals./1e6, 1));
pause
if opts.algtype ~= 2 % not EM
    [endvals, ~, minvals] = iterstats(opts.mfval(di,ci), 1);
    fprintf('\nmean(mfval): 1e6\n');
    fprintf('    %.4f', mean(endvals./1e6, 1));
    fprintf('\n');
    fprintf('    %.4f', mean(minvals./1e6, 1));
    pause
end
if isfield(opts, 'postdim')
    [endvals, ~, inivals] = iterstats(opts.postdim(di,ci), 1, @(x)x(1));
    fprintf('\nmean(postdim):\n');
    fprintf('     %4d', round(mean(endvals(:,1), 1)));
    fprintf('      %4d', round(mean(endvals(:,2:end), 1)));
    fprintf('\n');
    fprintf('     %4d', round(mean(inivals(:,1), 1)));
    fprintf('      %4d', round(mean(inivals(:,2:end), 1)));
    pause
end
if isfield(opts, 'etimes')
    [~, ~, totime] = iterstats(opts.etimes(di,ci), 1, @(x)sum(x));
    fprintf('\nmean(totime): (s)\n');
    fprintf('    %.4f', mean(totime, 1));
    fprintf('\n');
    pause
end
fprintf('\n');

[~, optcid] = min(mean(minerrs));
maxiter = max(iters(:,optcid));
optcid = ci(optcid);
figure;
plot(cell2mat(cellfun(@(x)[x,nan*ones(1,maxiter-numel(x))], opts.fvals(di,optcid), 'UniformOutput', false))', ...
    '*--', 'LineWidth', 1.5);
hold on;
if opts.algtype ~= 2 % not EM
    plot(cell2mat(cellfun(@(x)[x,nan*ones(1,maxiter-numel(x))], opts.mfval(di,optcid), 'UniformOutput', false))', ...
        '.-', 'LineWidth', 1.5);
end
pause
figure;
plot(cell2mat(cellfun(@(x)[x(1,:),nan*ones(1,maxiter-size(x,2))], opts.errs(di,optcid), 'UniformOutput', false))', ...
    '*--', 'LineWidth', 1.5);
hold on;
if opts.algtype ~= 2 % not EM
    plot(cell2mat(cellfun(@(x)[x(1,:),nan*ones(1,maxiter-size(x,2))], opts.merr(di,optcid), 'UniformOutput', false))', ...
        '.-', 'LineWidth', 1.5);
end
pause

close all

end