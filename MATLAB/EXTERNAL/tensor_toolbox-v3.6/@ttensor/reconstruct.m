function X = reconstruct(T,varargin)
%RECONSTRUCT Reconstruct or partially reconstruct tensor from ttensor.
%
%   X = RECONSTRUCT(T,K,SK) subsamples mode-K according to SK, which is
%   either a list of indices in the range {1,...,NK} or an MK x NK sampling
%   matrix where NK = NDIMS(T,K). This is much faster than constructing the
%   full tensor and then subsampling.
%
%   X = RECONSTRUCT(T,K1,S1,K2,S2,...) specifies multiple modes and
%   corresponding subsampling matrices or indices.  
%
%   X = RECONSTRUCT(T,{S1,S2,...SD}) specifies samplings for all modes. If
%   mode K is not downsampled, then set SK = []. 
%
%   <a href="matlab:web(strcat('file://',fullfile(getfield(what('tensor_toolbox'),'path'),'doc','html',ttensor_reconstruct_doc.html')))">Documentation page for ttensor/reconstruct</a>
%
%   See also HOSVD, TUCKER_ALS, TTENSOR.

d = ndims(T);
fsz = size(T);

%% Process inputs
if isempty(varargin)
    X = full(T);
    return;
end
if length(varargin) == 1
    if iscell(varargin)
        S = varargin{1};
    else
        error('X = RECONSTRUCT(T,{S1,S2,...SD}) requires second argument to be a cell array');
    end
else
    S = cell(d,1);
    for k = 1:d
        S{k} = [];
    end
    dd = length(varargin)/2;
    if dd ~= floor(dd)
        error('X = RECONSRUCT(T,K1,S1,K2,S2,...) requires and odd number of inputs');
    end
    for j = 1:dd
       k = varargin{2*j-1};
       S{k} = varargin{2*j};
    end
end

%% Reduce factor matrices

for k = 1:d
    if isempty(S{k})
        % Do nothing
    elseif ismatrix(S{k}) && (size(S{k},2)==fsz(k))
        T.u{k} = S{k} * T.u{k};
    else
        Uk = T.u{k};
        T.u{k} = Uk(S{k},:);
    end
end

X = full(T);
end