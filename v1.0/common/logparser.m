function [etime, fobjs] = logparser(fname, regvals, maxiter, nsubstep, regprec)
if ~exist('regprec', 'var')
    regprec = 4;
end
regvals = round(regvals * (10^regprec))/(10^regprec);
logfile = fopen(fname, 'r');
strptns = {
    '%f: %f (%fs) %*s';
    'di = %d';
    'C = %f';
%    '(%d,%d,%d): dy = %*f, dU = %*f, dV = %*f, dtheta = %*f, df = %f'
    '%s';
};
gcdi = 1;
cdi = ones(1, numel(regvals));
citer = ones(1, numel(regvals));
etime = zeros(3, numel(regvals), maxiter*nsubstep);
fobjs = zeros(size(etime));

while true
    tline = fgetl(logfile);
    if ~ischar(tline)
        break
    end
    for i = 1:numel(strptns)
        A = sscanf(tline, strptns{i});
        if ~isempty(A)
            break;
        end
    end
    switch i
        case 1,
            regvalid = find(A(1) == regvals);
            etime(cdi(regvalid), regvalid, citer(regvalid)) = A(3);
            fobjs(cdi(regvalid), regvalid, citer(regvalid)) = A(2);
            citer(regvalid) = citer(regvalid) + 1;
        case 2,
            gcdi = A;
        case 3,
            regvalid = find(regvals == round(A(1)*(10^regprec))/(10^regprec));
            cdi(regvalid) = gcdi;
            citer(regvalid) = 1;
        otherwise, 
    end
end
fclose(logfile);

end