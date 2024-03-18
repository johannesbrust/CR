function Y = scale(X,S,dims,func)
%SCALE Scale along specified dimensions of tensor.
%
%   Y = SCALE(X,S,DIMS) scales the tensor X along the dimension(s)
%   specified in DIMS using the scaling data in S. If DIMS contains
%   only one dimension, then S can be a column vector. Otherwise, S
%   should be a tensor.
%
%   Y = SCALE(X,S,DIMS,FN) modifies the tensor X along the dimension(s)
%   specified in DIMS using the data in S and the function specified by FN.
%   The function FN should take two arguments, the first being the tensor
%   data and the second being the scaling data, to be combined using
%   bsxfun. If DIMS contains only one dimension, then S can be a column
%   vector. Otherwise, S should be a tensor. 
%
%   Examples
%   X = tenones([3,4,5]);
%   S = 10 * [1:5]'; Y = scale(X,S,3)
%   S = tensor(10 * [1:5]',5); Y = scale(X,S,3)
%   S = tensor(1:12,[3 4]); Y = scale(X,S,[1 2])
%   S = tensor(1:12,[3 4]); Y = scale(X,S,-3)
%   S = tensor(1:60,[3 4 5]); Y = scale(X,S,1:3)
%
%   X = tensor(1:24,[4 3 2]); %<-- Generate some data.
%   mu = collapse(X,2,@mean); %<-- Calculate means of mode-2 fibers
%   Y = scale(X,mu,[1 3],@(x,y) x-y); %<-- Center mode-2 fibers
%   mu_new = collapse(Y,2,@mean) %<-- Mode-2 fibers have mean zero
%
%   <a href="matlab:web(strcat('file://',fullfile(getfield(what('tensor_toolbox'),'path'),'doc','html','collapse_scale_doc.html')))">Documentation page for collapsing and scaling tensors</a>
%
%   See also TENSOR, TENSOR/COLLAPSE.
%
%Tensor Toolbox for MATLAB: <a href="https://www.tensortoolbox.org">www.tensortoolbox.org</a>

dims = tt_dimscheck(dims,ndims(X));
remdims = setdiff(1:ndims(X),dims);

if ~exist('func','var') 
    ver = 2;        % New version
    func = @times;  % Default function
elseif isa(func,'function_handle')
    ver = 2; % New version with user-provided function
else
    ver = 1; % Explicitly use old version, useful for testing
end

% Convert to a matrix so that each column of A can be scaled by a
% vectorized version of S.
A = double(tenmat(X,dims,remdims));

switch(class(S))
    case {'tensor'}
        if ~isequal(size(S), X.size(dims))
            error 'Size mismatch';
        end
        % Vectorize S.
        S = double(tenmat(S,1:ndims(S),[]));
    case {'double'}
        if size(S,1) ~= X.size(dims)
            error 'Size mismatch';
        end
    otherwise
        error('Invalid scaling factor');
end

[m,n] = size(A);

% If the size of S is pretty small, we can convert it to a diagonal matrix
% and multiply by A. Otherwise, we scale A column-by-column.
if ver == 1
    if (m <= n)
        B = diag(S) * A;
    else
        B = zeros(size(A));
        for j = 1:n
            B(:,j) = S .* A(:,j);
        end
    end
else % ver == 2
    B = bsxfun(func,A,S);
end

% Convert the matrix B back into a tensor and return.
Y = tensor(tenmat(B,dims,remdims,X.size));




