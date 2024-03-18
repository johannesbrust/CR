function Xvec = vec(X)
%VEC Vectorize a tensor.
%
%   XVEC = VEC(X) returns the vectorization of X as a standard MATLAB
%   array.
%
%   See also TENSOR, TENSOR/UNFOLD, TENMAT.
%
%Tensor Toolbox for MATLAB: <a href="https://www.tensortoolbox.org">www.tensortoolbox.org</a>

Xvec = X.data(:);
