% calculate the expected MAE of random guess,
% as used to normalize MAE to get NMAE
%
% L - rating level (5 for MovieLens, 6 for EachMovie)
%
% Written by Minjie Xu (chokkyvista06@gmail.com)

function y = emae(L)
y = sum((2:L).*(1:(L-1)))/L^2;

end