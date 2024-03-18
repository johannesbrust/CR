function [fw,gw] = logobji(w,X,yi)
%LOGOBJi Computes the function and gradient for one data pair
%   
%   This implementation reshapes w and x to avoid loops
% 
%--------------------------------------------------------------------------
% 01/19/24, J.B., initial implementation

nn      = size(X,1);
nn2     = nn*nn;
x       = reshape(X,[nn2,1]);
W       = reshape(w,[10,nn2]);
Wx      = W*x;

xWx     = exp(Wx);
sxWx    = sum(xWx);

fw      = log(sxWx) - Wx(yi+1);


% gw      = kron((xWx./sxWx),x);
% 
% %fprintf('yi*nn2 = %i \n', yi*nn2);
% 
% idx     = yi*nn2 + (1:nn2);
% 
% %
% % For debugging
% %
% % fprintf('size(gw) = %i \n', size(gw));
% % fprintf('size(x) = %i \n', size(x));
% % fprintf('length(idx) = %i \n', length(idx));
% 
% gw(idx) = gw(idx) - x;


%
% Modified derivative
%

xWxh        = xWx / sxWx ;
xWxh(yi+1)  = xWxh(yi+1) - 1;

gw = kron(x,xWxh);

end