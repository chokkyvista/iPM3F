% Calculation of predicted labels for MMMF given soft outputs and thresholds
%
% function [y] = m3fSoftmax(xy,theta)
% xy - matrix of soft (real-valued) outputs [n,m]
% theta - matrix of thresholds (one vector per row) [n,l-1]
% y - matrix of hard (1...l) outputs [n,m]
%
% Written by Jason Rennie, February 2005
% Last modified: Sun Dec 17 22:28:17 2006

function [y] = m3fSoftmax(xy,theta)
  [n,m] = size(xy);
  [n1,l1] = size(theta);
  if n ~= n1
    error('sizes of xy and theta don''t match');
  end
  y = ones(n,m);
  for i=1:l1
    y = y + (xy >= theta(:,i*ones(1,m))); % modified: Minjie Xu
  end
