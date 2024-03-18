function [fsampler, fsampler_str] = tt_fsampler_setup(X, varargin)
%TT_FSAMPLER_SETUP sets up random sampling of the elements of a tensor.
%
%   FSAMPLER = TT_FSAMPLER_SETUP(X) will return a set of elements using
%   stratified sampling if the tensor is sparse and uniform sampling if the
%   tensor is dense. The number of samples drawn is based off either the
%   number of nonzeros (sparse) or the size of the tensor (dense).
%
%   FSAMPLER = TT_FSAMPLER_SETUP(X, 'fsamp', FSAMP) specifies the number of
%   elements to draw.  For sparse tensors, it will draw FSAMP non-zeros and
%   FSAMP zeros for a total of 2 * FSAMP elements.  For dense tensors, it
%   will draw FSAMP elements uniformly.
%
%   FSAMPLER = TT_FSAMPLER_SETUP(X, 'fsampler_type', TYPE, 'fsamp', FSAMP) 
%   allows the user to specify which method should be used via TYPE:
%       'stratified' - Stratified sampling with 2 * FSAMP total elements
%       'uniform' - Uniform sampling with FSAMP elements
%   If FSAMP is left empty, the default sample number will be used.  If
%   TYPE is a function handle, this will be assumed to be a user-specified
%   FSAMPLER and the handle will simply be returned.
%
%   FSAMPLER = TT_FSAMPLER_SETUP(X,...,'xnzidx', XNZIDX) allows the user to
%   provide the list of nonzero indices for stratified sampling to avoid
%   duplicate computation.
%
%   The default setting for FSAMP is 4 percent of the nonzeros for
%   stratified sampling and 10 percent of the tensor elements for uniform
%   sampling.
%
%   REFERENCES: 
%   * C. Battaglino, G. Ballard, T. G. Kolda, A Practical Randomized CP
%     Tensor Decomposition, SIAM J. Matrix Analysis and Applications, 
%     39(2):876-901, 2018, https://doi.org/10.1137/17M1112303.
%   * T. G. Kolda, D. Hong, Stochastic Gradients for Large-Scale Tensor
%     Decomposition. SIAM J. Mathematics of Data Science, 2:1066-1095,
%     2020, https://doi.org/10.1137/19m1266265
%
%Tensor Toolbox for MATLAB: <a href="https://www.tensortoolbox.org">www.tensortoolbox.org</a>

% Created by Tamara G. Kolda, Fall 2018. Includes work with
% collaborators David Hong and Jed Duersch. 
% This code was adapted from the fsampler in gcp_opt.m.  The oversamp
% parameter was removed from the call to tt_stratified_sampling.  It will
% simply use the default value of 1.1.

% Process inputs
params = inputParser;
params.addParameter('fsamp',[]);
params.addParameter('fsampler_type', []);
params.addParameter('xnzidx', []);
params.parse(varargin{:});

fsamp = params.Results.fsamp;
fsampler_type = params.Results.fsampler_type;
xnzidx = params.Results.xnzidx;

issparse = isa(X,'sptensor');
sz = size(X);
tsz = prod(sz);

if isempty(fsampler_type)
    if issparse
        fsampler_type = 'stratified';
    else
        fsampler_type = 'uniform';
    end
end
    
if isa(fsampler_type,'function_handle')
    
    fsampler = fsampler_type;
    fsampler_str = 'user-specified';
    
elseif strcmp(fsampler_type, 'stratified')
    
    nnonzeros = nnz(X);
    if isempty(fsamp) 
        nzeros = tsz - nnonzeros;
        % Default fsamp is 4 percent of nonzeros
        ftmp = max(ceil(nnonzeros/25), 10^5);
        fsamp(1) = min(ftmp, nnonzeros);
        fsamp(2) = min([ftmp, nnonzeros, nzeros]);
    elseif length(fsamp) == 1
        % Warn user if fsamp seems high
        if (fsamp > (nnonzeros/4))
            fprintf('Warning: nsampfit is greater than 25 percent of the nonzero elements. \n');
            fprintf('This will likely result in suboptimal performance due to long fit times. \n');
            fprintf('\n');
        end
        tmp = fsamp;
        fsamp(1) = tmp;
        fsamp(2) = tmp;
    end


    
    % Create and sort linear indices of X nonzeros for the sampler
    if isempty(xnzidx)
        xnzidx = tt_sub2ind64(sz,X.subs);
        xnzidx = sort(xnzidx);
    end
    
    fsampler = @() tt_sample_stratified(X, xnzidx, fsamp(1), fsamp(2));
    fsampler_str =  sprintf('stratified with %d nonzero and %d zero samples', fsamp);
    
    
elseif strcmp(fsampler_type, 'uniform')
    
    if isempty(fsamp)
        fsamp = min( max(ceil(tsz/10), 10^6), tsz ); 
    elseif (fsamp > (tsz/2))
        % Warn user if fsamp seems high
        fprintf('Warning: nsampfit is greater than 50 percent of the tensor elements. \n');
        fprintf('This will likely result in suboptimal performance due to long fit times. \n');
        fprintf('\n');
    end
    
    fsampler = @() tt_sample_uniform(X,fsamp);
    fsampler_str = sprintf('uniform with %d samples', fsamp);
    
else
    
    error('Invalid choice for ''fsampler''');
    
end