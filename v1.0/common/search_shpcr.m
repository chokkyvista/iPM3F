% function [x1, x2] = search_shpcr(deszcrs, aszcrs, C, rhor, varsigma)
function [x1, x2] = search_shpcr(zcrs, slope, C, rg, hinvrg)
% search for the optimal solution to an objective of the form:
%     C*(sum of hinge loss) + STRICTLY convex regularizer
%       's'    'h'         'p'        'c'    'r'
% C: regularization constant (must be positive)
% rg: gradients of the regularizer on points [-inf, zcrs]
%     e.g., in SVMs, rg = [-inf, zcrs];
% hinvrg: function handle to the inverse gradient

tpos = slope~=0;
zcrs = zcrs(tpos);
slope = slope(tpos);
if ~exist('C', 'var')
    C = 1;
    rg = [0, zeros(size(zcrs))];
    lossonly = true;
else
    lossonly = false;
    if ~exist('rg', 'var')
        rg = [-inf, zcrs];
        hinvrg = @invrg_svm;
    else
        rg = rg([true, tpos]);
    end
end

shg_lm = sum(slope(slope<0));
[zcrs, tpos] = sort(zcrs, 'ascend');
shg = cumsum([shg_lm, abs(slope(tpos))]);
rg = [rg(1), rg(tpos+1)];

%           _____|_____|......|_____|_____|......|_____
% zcrs:          1     2 ... k-1    k    k+1 ... N
% shg:        1     2    ...     k    k+1    ...   N+1
% rg:      1     2     3 ...  k    k+1   k+2... N+1
% objgl:    1     2     3 ...  k    k+1   k+2... N+1
% objgl?0: <=    <=    <=     <=     >     >      >
objgl = C*shg + rg;
if objgl(1) > 0, x1 = -inf; x2 = x1; % impossible the case when 'lossonly'
elseif objgl(1) == 0
    x1 = -inf;
    if lossonly, x2 = zcrs(1);
    else x2 = x1; end;
elseif objgl(end) < 0, x1 = hinvrg(-C*shg(end)); x2 = x1; % impossible the case when 'lossonly'
elseif objgl(end) == 0
    x1 = zcrs(end);
    if lossonly, x2 = inf;
    else x2 = x1; end;
else
%     k = find(objgl>0, 1, 'first') - 1;
    k = ismembc2(false, objgl>0); % last pos where objgl<=0; k<end
    if lossonly
        if (objgl(k)<0)
            x1 = zcrs(k); x2 = x1;
        else
            x2 = zcrs(k);
            if k > 1, x1 = zcrs(k-1);
            else x1 = -inf; end
        end
    else
        if C*shg(k)+rg(k+1) <= 0, x1 = zcrs(k); x2 = x1;
        else x1 = hinvrg(-C*shg(k)); x2 = x1; end
    end
end

end

function y = invrg_svm(x)
y = x;
end

% tst = 0; tss = 0;
% for i = 1:10000
%     deszcrs = randn(1, randi(500))*100+randi(100);
%     aszcrs = randn(1,randi(500))*100+randi(100);
%     C = abs(randn*10);
%     rhor = rand*max(abs([deszcrs, aszcrs]))*(2*(rand>0.5)-1);
%     varsigma = std([deszcrs, aszcrs], 1)*2*rand;
%     tStart = tic;[x1, x2] = search_theta(deszcrs, aszcrs, C, rhor, varsigma);tst = tst + toc(tStart);
%     zcrs = [deszcrs, aszcrs];
%     slope = [-ones(1,numel(deszcrs)),ones(1,numel(aszcrs))];
%     rg = ([-inf,zcrs]-rhor)./varsigma^2;
%     tStart = tic;[y1, y2] = search_shpcr(zcrs, slope, C, rg, @(x)varsigma^2*x+rhor);tss = tss + toc(tStart);
%     if abs(y1-x1)>1e-10 || abs(y2-x2)>1e-10
%         fprintf('BUG!\n');
%         break;
%     end
% end
% 
% tst = 0; tss = 0;
% for i = 1:10000
%     deszcrs = randn(1, randi(500))*100+randi(100);
%     aszcrs = randn(1,randi(500))*100+randi(100);
%     tStart = tic;[x1, x2] = search_theta(deszcrs, aszcrs);tst = tst + toc(tStart);
%     zcrs = [deszcrs, aszcrs];
%     slope = [-ones(1,numel(deszcrs)),ones(1,numel(aszcrs))];
%     tStart = tic;[y1, y2] = search_shpcr(zcrs, slope);tss = tss + toc(tStart);
%     if y1~=x1 || y2~=x2
%         fprintf('BUG!\n');
%         break;
%     end
% end
