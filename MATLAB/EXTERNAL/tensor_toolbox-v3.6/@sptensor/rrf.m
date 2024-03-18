function U = rrf(X,k,Xfidx,R,s)
%RRF Produce matrix via sparse randomized range finder in mode-k.
%
%   U = RRF(X,K,XFIDX,R,S) selects S random columns of the mode-k
%   unfolding of X and randomly combines those to form an NK x R matrix.
%   Here, XFIDX is the mode-K fiber indices of the nonzeros in X, i.e., 
%
%     XFIDX = FINDICES(X,K);
%
%   The resulting matrix U is of size NK x R where NK = SIZE(X,k). This is
%   useful as an initial guess for computing a CP decomposition.
%
%   See also FINDICES, CP_ARLS_LEV.
%
%Tensor Toolbox for MATLAB: <a href="https://www.tensortoolbox.org">www.tensortoolbox.org</a>


%%
sz = size(X); % Tensor dimensions

%% Sample non-zero fibers
Xfidx_unique = unique(Xfidx);
fidx = Xfidx_unique(randi(length(Xfidx_unique), s, 1));

%% Extract fibers
[fidx_unique, ~, unique2repeat] = unique(fidx);

% Extract the row index and value for every nonzero whose mode-k fiber
% index that is in the sample set 
[tf,loc] = ismember(Xfidx, fidx_unique);

ii_unique = X.subs(tf, k); % Extract the mode-k index
jj_unique = loc(tf); % Extract the mode-k fiber index
vv_unique = X.vals(tf); % Extract the corresponding value
jj_unique = double(jj_unique);
Xsamp_unique = sparse(ii_unique, jj_unique, vv_unique, sz(k), length(fidx_unique));

% Convert back
Xks = Xsamp_unique(:,unique2repeat);


%% Form random linear combination
U = Xks * randn(s, R);
