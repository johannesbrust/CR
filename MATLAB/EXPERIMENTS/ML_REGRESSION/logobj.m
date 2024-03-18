function [fw,gw] = logobj(w,X,y)
%LOGOBJ Computes the function and gradient for the logistic loss
%   
%   This function enables batch evaluations by calling 
%   logobji(w,Xi,yi) for each data pair
% 
%--------------------------------------------------------------------------
% 01/20/24, J.B., initial implementation

%
% Initialize dimensions, function and gradient
%
d   = size(w,1);
fw  = 0;
gw  = zeros(d,1);
N   = length(y);

%
% Accumulate relevant values
%
for i = 1:N

    [fwi,gwi]   = logobji(w,X(:,:,i),y(i));
    fw          = fw + fwi;
    gw          = gw + gwi;

end

%
% Scale quantities
%
fw = fw./ N;
gw = gw./ N;

end