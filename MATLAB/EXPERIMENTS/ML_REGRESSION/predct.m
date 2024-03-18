function [z] = predct(w,X)
%predct Makes a prediction based on trained weights w for data X
%   
%   [z] = predict(w,X) computes an nonnegative vector of approximate 
%   probabilies. For instance, z(1) represents the probability that
%   the data belongs to class 1.
% 
%--------------------------------------------------------------------------
% 01/20/24, J.B., initial implementation

nn      = size(X,1);
nn2     = nn*nn;
x       = reshape(X,[nn2,1]);
W       = reshape(w,[10,nn2]);
Wx      = W*x;

xWx     = exp(Wx);
sxWx    = sum(xWx);

z       = xWx./ sxWx;

end