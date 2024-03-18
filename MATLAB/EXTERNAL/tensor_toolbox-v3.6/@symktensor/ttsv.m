function y = ttsv(A,x,n)
%TTSV Symktensor times same vector in multiple modes.
%
%   Y = TTSV(A,X) multiples the symktensor A by the vector X in all modes.
%
%   Y = TTSV(A,X,-1) multiplies the symktensor A by the vector X in all modes
%   but the first. Returns the answer as a normal MATLAB array (not a
%   tensor). 
%
%   Y = TTSV(A,X,-2) multiplies the symktensor A by the vector X in all modes
%   but the first two. Returns the answer as a normal MATLAB matrix (not a
%   tensor). 
%
%   Y = TTSV(A,X,-N) multiplies the symktensor A by the vector X is all modes
%   but the first N. Returns the answer as a symktensor.
%
%   See also TENSOR, TENSOR/TTSV.
%
%Tensor Toolbox for MATLAB: <a href="https://www.tensortoolbox.org">www.tensortoolbox.org</a>

if ~exist('n','var')
    n = 0;
elseif (n > 0)
    error('Invalid usage');
end

lambda=A.lambda;
v=A.u;
m=A.m;

if(n==0)
    y=lambda'*((v'*x).^m);
elseif (n==-1)
        y=v*diag(lambda)*((v'*x).^(m-1));
elseif (n==-2)
        y=v*diag(lambda)*diag((v'*x).^(m-2))*v';
else
        lambda_y=lambda.* (v'*x).^(m+n);
        y=symktensor(lambda_y,v,-n);
end

end