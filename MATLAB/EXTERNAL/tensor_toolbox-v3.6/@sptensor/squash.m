function [Y,S] = squash(X)
%SQUASH Remove empty slices from a sparse tensor.
%
%   Y = SQUASH(X) returns a sparse tensor Y with the same elements as
%   X but with all the empty slices removed.  The indices appearing in the
%   tensor X for each dimension n are remapped to the range [1:M_n] where M_n
%   is the number of unique indices for dimension n.
%
%   [Y,S] = squash(X) also returns a cell-array of length ndims(X) which
%   specifies which indices in X the indices in Y correspond to, i.e.,
%   X.subs(:,n) == S{n}(Y.subs(:,n)) for each n.
%
%   See also SPTENSOR.
%
%Tensor Toolbox for MATLAB: <a href="https://www.tensortoolbox.org">www.tensortoolbox.org</a>

if ~isa(X,'sptensor')
  error('Input must be an sptensor!');
end

d = ndims(X);
subs = zeros(size(X.subs));
sz = zeros(1,d);
if nargout == 2
  S = cell(d,1);
end
for n=1:d
  [s,~,j] = unique(X.subs(:,n));
  subs(:,n) = j;
  sz(n) = length(s);
  if nargout == 2
    S{n} = s;
  end
end

Y = sptensor(subs, X.vals, sz);
