function Xmat = unfold(X,rdims,cdims)
%UNFOLD Unfold a tensor into a matrix.
%
%   XMAT = UNFOLD(X,K) return the mode-K unfolding of X as a standard
%   MATLAB matrix object. 
%
%   XMAT = UNFOLD(X,RDIMS,CDIMS) returns the unfolding of X with the modes
%   in RDIMS mapped to the rows and modes in CDIMS mapped to the columns. 
%
%   See also TENSOR, TENSOR/VEC, TENMAT.
%
%Tensor Toolbox for MATLAB: <a href="https://www.tensortoolbox.org">www.tensortoolbox.org</a>


sz = size(X);
d = ndims(X);
if ~exist('rdims','var')
    Xmat = X.data(:);
    return;
end
if ~exist('cdims','var')    
    tmp = true(1,d);
    tmp(rdims) = false;
    if length(tmp) > d
        error('unfolding dimensions out of range');
    end
    cdims = find(tmp); % Faster than: cdims = setdiff(1:d, rdims);
else
    tmp = true(1,d);
    tmp(rdims) = false;
    tmp(cdims) = false;
    if length(tmp) > d
        error('unfolding dimensions out of range');
    end
    if any(tmp)
        error('not all dimensions have been mapped for unfolding')
    end
end
Xmat = reshape(permute(X.data,[rdims cdims]), prod(sz(rdims)), prod(sz(cdims)));
