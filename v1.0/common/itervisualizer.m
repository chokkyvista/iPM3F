function [totime, mfobjs] = itervisualizer(etime, fobjs, didx, cidx, iidx, distype)
if nargin < 6
    distype = 2;
end
if numel(didx)==1 || numel(cidx)==1
    figure;
    ppfobjs = squeeze(fobjs(didx,cidx,iidx))';
    ppcsetime = squeeze(cumsum(etime(didx,cidx,iidx), 3))';
    switch distype
        case 0,
            plot(ppfobjs);
        case 1,
            plot(ppcsetime);
        case 2,
            plot(ppfobjs/max(ppfobjs(:)));
            hold on;
            plot(ppcsetime/max(ppcsetime(:)));
            hold off;
%             plotyy(iidx, squeeze(fobjs(didx,cidx,iidx))', iidx, squeeze(cumsum(etime(didx,cidx,iidx), 3))');
    end
    
    if numel(didx)==1
        legend(num2str(cidx(:), '%d'));
    else
        legend(num2str(didx(:), '%d'));
    end
    
    totime = ppcsetime(end,:);
else
    totime = squeeze(sum(etime(didx,cidx,iidx), 3));
end

mfobjs = zeros(numel(didx), numel(cidx));
for di = 1:numel(didx)
    for ci = 1:numel(cidx)
        mfidx = find(squeeze(fobjs(didx(di),cidx(ci),iidx)), 1, 'last');
        if isempty(mfidx)
            mfobjs(di, ci) = nan;
        else
            mfobjs(di, ci) = fobjs(didx(di),cidx(ci),mfidx);
        end
    end
end

end