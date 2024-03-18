function [leveragescores] = tt_leverage_scores(A)
%TT_LEVERAGE_SCORES computes the leverage scores of matrix.
%
%   L = TT_LEVERAGE_SCORES(A) computes the leverage scores of a matrix A.
%   The leverage scores are the row norms of the left singular vectors
%   of A. They are between 0 and 1 inclusive and measure the fractional
%   contribution of a row to the column space of A.  The length of L is the
%   number of rows of A.  
%
%   See also CP_ARLS_LEV, TT_SAMPLED_SOLVE.
%
%Tensor Toolbox for MATLAB: <a href="https://www.tensortoolbox.org">www.tensortoolbox.org</a>

[V, ~, ~] = svd(A, 'econ');
leveragescores = sum(V.^2, 2);
end