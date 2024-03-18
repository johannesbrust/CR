function V = mttkrp(X,U,n,nzchunk,rchunk,ver)
%MTTKRP Matricized tensor times Khatri-Rao product for sparse tensor.
%
%   NOTICE: This internals of this code changed in Version 3.3 of Tensor
%   Toolbox to be much more efficient. It now "chunks" the nonzeros as well
%   as the factor matrices. Special options for this are described below.
%
%   V = MTTKRP(X,U,N) efficiently calculates the matrix product of the
%   n-mode matricization of X with the Khatri-Rao product of all
%   entries in U, a cell array of matrices, except the Nth.  How to
%   most efficiently do this computation depends on the type of tensor
%   involved.
%
%   V = MTTKRP(X,K,N) instead uses the Khatri-Rao product formed by the
%   matrices and lambda vector stored in the ktensor K. As with the cell
%   array, it ignores the Nth factor matrix. The lambda vector is absorbed
%   into one of the factor matrices.
%
%   V = MTTKRP(X,U,N,0) reverts to the OLD version of MTTKRP prior to
%   Tensor Toolbox Version 3.3, which repeatedly calls TTV.
%
%   V = MTTKRP(X,U,N,NZCHUNK,RCHUNK) specifies the "chunk" sizes for the
%   nonzeros and factor matrix columns, respectively. These default to
%   NZCHUNK=1e4 and RCHUNK=10 if not specified. If NZCHUNK=NNZ(X) and
%   RCHUNCK=SIZE(U{1},2), then it's just one big chunk.
%
%   V = MTTKRP(X,U,N,NZCHUNK,RCHUNK,2) swaps the loop order so that the
%   R-loop is INSIDE the NZ-loop rather than the reverse, which is the
%   default.
%
%   Examples
%   S = sptensor([3 3 3; 1 3 3; 1 2 1], 4, [3, 4, 3]); %<-Declare sptensor
%   mttkrp(S, {rand(3,3), rand(3,3), rand(3,3)}, 2)
%
%   See also TENSOR/MTTKRP, SPTENSOR/TTV, SPTENSOR
%
%Tensor Toolbox for MATLAB: <a href="https://www.tensortoolbox.org">www.tensortoolbox.org</a>

% In the sparse case, we do not want to form the Khatri-Rao product.

N = ndims(X);

if isa(U,'ktensor')
    % Absorb lambda into one of the factors, but not the one that's skipped
    if n == 1
        U = redistribute(U,2);
    else
        U = redistribute(U,1);
    end    
    % Extract the factor matrices
    U = U.u;
end

if (length(U) ~= N)
    error('Cell array is the wrong length');
end

if ~iscell(U)
    error('Second argument should be a cell array or a ktensor');
end

if (n == 1)
    R = size(U{2},2);
else
    R = size(U{1},2);
end

if ~exist('nzchunk','var')
    nzchunk = 1e4;
end
if ~exist('rchunk','var')
    rchunk = 10;
end
if ~exist('ver','var')
    if nzchunk <= 0
        ver = 0;
    else
        ver = 1;
    end
end


if ver == 0 % OLD WAY
    
    V = zeros(size(X,n),R);

    for r = 1:R
        % Set up cell array with appropriate vectors for ttv multiplication
        Z = cell(N,1);
        for i = [1:n-1,n+1:N]
            Z{i} = U{i}(:,r);
        end
        % Perform ttv multiplication
        V(:,r) = double(ttv(X, Z, -n));
    end
    
elseif ver == 1 % NEW DEFAULT 'CHUNKED' APPROACH
    
    nz = nnz(X);
    d = ndims(X);
    nn = size(X,n);

    V = zeros(nn,R);    
    rctr = 0;
    while (rctr < R)
        
        % Process r range from rctr1 to rctr (columns of factor matrices)
        rctr1 = rctr + 1;
        rctr = min(R, rctr + rchunk);
        rlen = rctr - rctr1 + 1;
        
        nzctr = 0;
        while (nzctr < nz)
            
            % Process nonzero range from nzctr1 to nzctr 
            nzctr1 = nzctr+1;
            nzctr = min(nz,nzctr1+nzchunk);   
            
            % ----
            Vexp = repmat(X.vals(nzctr1:nzctr),1,rlen);
            for k = [1:n-1, n+1:d]
                Ak = U{k};
                Akexp = Ak(X.subs(nzctr1:nzctr,k),rctr1:rctr);
                Vexp = Vexp .* Akexp;
            end
            for j = rctr1:rctr
                vj = accumarray(X.subs(nzctr1:nzctr,n), Vexp(:,j-rctr1+1), [nn 1]);
                V(:,j) = V(:,j) + vj;
            end
            % ----
        end
    end

elseif ver == 2 % 'CHUNKED' SWAPPING R & NZ CHUNKS
    
    nz = nnz(X);
    d = ndims(X);
    nn = size(X,n);

    V = zeros(nn,R);    
    nzctr = 0;
    while (nzctr < nz)
        
        % Process nonzero range from nzctr1 to nzctr
        nzctr1 = nzctr+1;
        nzctr = min(nz,nzctr1+nzchunk);

        rctr = 0;
        Xvals = X.vals(nzctr1:nzctr);
        while (rctr < R)
            
            % Process r range from rctr1 to rctr (columns of factor matrices)
            rctr1 = rctr + 1;
            rctr = min(R, rctr + rchunk);
            rlen = rctr - rctr1 + 1;
            
            % ----
            Vexp = repmat(Xvals,1,rlen);
            for k = [1:n-1, n+1:d]
                Ak = U{k};
                Akexp = Ak(X.subs(nzctr1:nzctr,k),rctr1:rctr);
                Vexp = Vexp .* Akexp;
            end
            for j = rctr1:rctr
                vj = accumarray(X.subs(nzctr1:nzctr,n), Vexp(:,j-rctr1+1), [nn 1]);
                V(:,j) = V(:,j) + vj;
            end
            % ----

        end
    end
    
end
