% update the averaged sample in infinite latent feature models
% 
% mnX   - N*I, averaged sample with only the I still-active features
%         (before the latest iteration)
% X     - N*K, the latest single sample
% navg  - # of samples to average (inc. the latest one)
% zcidx - 1*I, logical vector, true for all-zero columns (the newly-
%         inactive features from the latest iteration)
% newK  - # of new features
% mzX   - N*J, averaged sample with only the J already-inactive 
%         features (before the latest iteration)
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function [mnX, mzX] = updatem(mnX, X, navg, zcidx, newK, mzX)
if exist('mzX', 'var')
    mzX = mzX.*(1-1/navg); mzX = [mzX, mnX(:,zcidx)./navg];
end
mnX = [mnX(:,~zcidx),zeros(size(X,1),newK)].*(1-1/navg) + X./navg;

end
