function [x1, x2] = search_theta(deszcrs, aszcrs, C, rhor, varsigma)
% objective function to minimize:
% f = @(x)(sum(max(0,repmat(deszcrs',1,numel(x))-x(ones(1,numel(deszcrs)),:)))+sum(max(0,x(ones(1,numel(aszcrs)),:)-repmat(aszcrs',1,numel(x)))))*C+0.5/varsigma^2*(x-rhor).^2

if ~exist('rhor', 'var')
    rhor = 0;
end
if ~exist('varsigma', 'var')
    lambda = 0;
else
    lambda = 1/(C*varsigma^2);
end
deszcrs = sort(deszcrs(:), 'ascend')';
aszcrs = sort(aszcrs(:), 'ascend')';
allzcrs = unique([deszcrs, aszcrs]); % 'unique' sorts output in ascending order

nzcrs = numel(allzcrs);
allzcrs = [allzcrs(1)-1, allzcrs, allzcrs(end)+1];
a = 2;
b = nzcrs + 1;
while a <= b
    i = ceil((a+b)/2);
    x = allzcrs(i);
    xl = (x + allzcrs(i-1)) * 0.5;
    deltal = sum(deszcrs > xl) - sum(aszcrs < xl) - lambda*(x-rhor);
    xr = (x + allzcrs(i+1)) * 0.5;
    deltar = sum(aszcrs < xr) - sum(deszcrs > xr) + lambda*(x-rhor);
    if deltal >= 0 && deltar >= 0
        break;
    elseif deltar < 0
        a = i + 1;
    else
        b = i - 1;
    end
end

if deltal == 0
    x2 = x;
    x1 = allzcrs(max(2,i-(lambda==0)));
elseif deltar == 0
    x1 = x;
    x2 = allzcrs(min(i+(lambda==0),nzcrs+1));
elseif deltal > 0 && deltar > 0
    x1 = x;
    x2 = x;
else % a = b + 1
    if deltal < 0
        x1 = x+deltal/lambda;
    else
        x1 = x-deltar/lambda;
    end
    x2 = x1;
end