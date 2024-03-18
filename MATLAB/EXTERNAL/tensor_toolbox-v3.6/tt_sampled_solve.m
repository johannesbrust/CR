function [V,info] = tt_sampled_solve(X,A,Alev,k,s,c,tau,Xfidx,calcres,combrep)
%TT_SAMPLED_SOLVE Sampled solve for CP-ARLS using leverage scores.
%
%   V = TT_SAMPLED_SOLVE(X,A,ALEV,K,S,C,XFIDX,CALCRES,COMPREF) solves a
%   sampled version of the CP least squares problem for mode-K with the
%   following inputs... 
%
%     * X = D-way tensor, which can be dense or sparse.
%     * A = cell array of D factor matrices     
%     * Alev = cell array of leverage scores for the factor matrices,
%       i.e., Alev{k} is a column vector of length N(k) where N = size(X).
%     * K = Specified mode for solve.
%     * S = Number of samples. Usually R < S << prod(N) where N = size(X).
%     * C = Upper bound on number of draws of single sample, in
%       expectation, use [] to do no damping       
%     * TAU = Theshold for deterministic inclusion, use [] to to include
%       no samples deterministically
%     * XFIDX = Mode-K fiber indices; use [] for dense tensor.
%     * CALCRES = 0 default, 1 to save out sovle info, 2 to also calculate 
%       residual of sampled system. 
%     * COMBREP = True to cobmine repeated rows, should always be set to true.
%
%   REFERENCE: B. Larsen T. G. Kolda. Practical Leverage-Based Sampling
%   for Low-Rank Tensor Decomposition, 2020.
%   https://arxiv.org/abs/2006.16438
%
%   See also TT_LEVERAGE_SCORES (to calculate Alev), FINDICES (to calcuate
%   XFIDX).
%
%Tensor Toolbox for MATLAB: <a href="https://www.tensortoolbox.org">www.tensortoolbox.org</a>


ss_overall_timer = tic;
%%
d = ndims(X);
n = size(X);
mrng = [1:k-1, k+1:d];

%% Sample multi-indices
ss_timer = tic;


if (isempty(tau))
    % Case 1: No deterministic inclusion
    [midx, prob_rnd, alpha] = sample_krp_leverage(Alev(mrng),s,c);
    wgts = 1./sqrt(s * prob_rnd);
    
    
    % Rename original midx to make difference clear
    % Combine repeated rows and scale by sqrt(cnt)
    if combrep
        [midx,IA,mm] = unique(midx,'row');
        midx_cnts = accumarray(mm,1);
        wgts = wgts(IA).*sqrt(midx_cnts);
    end
    
    % Save out empty values for the deterministic outputs
    s_det = 0;
    psum_det = 0;
    prob_det = [];
    midx_det = [];
else
    % Case 2: Deterministic inclusion above tau
    
    % Find deterministic samples and associated probabilities
    % lidx_det = linear indices of deterministic samples
    % sdet = number of deterministic samples
    % pdet = total probability of deterministic samples
    % pvec = individual probabilites (only used for extra information)
    [lidx_det, s_det, psum_det, prob_det] = find_deterministic_krp(Alev(mrng), tau, s);
    
    % Draw random samples and then reject any in the determinstic set.
    % Need to draw example samples to account for rejection.
    s_rnd = s - s_det; % Number of random samples to draw
    ovrsmpl = 10; % Oversample rate
    [midx_rnd, prob_rnd, alpha] = sample_krp_leverage(Alev(mrng), ovrsmpl*s_rnd, c);

    % Reject samples included in deterministic set (using linear indices)
    lidx_rnd = tt_sub2ind64(n(mrng), midx_rnd);
    idx_keep = ~ismember(lidx_rnd, lidx_det);
    midx_rnd = midx_rnd(idx_keep, :);
    prob_rnd = prob_rnd(idx_keep, :);

    % Compute number of samples actually achieved. It could be less
    % than s_rnd if too many samples are rejected.
    s_rnd = min(s_rnd, size(midx_rnd, 1));
        
    % Throw out any extra samples
    midx_rnd = midx_rnd(1:s_rnd, :);
    prob_rnd = prob_rnd(1:s_rnd, :);

    % Rescale the probability to account for ignoring determinstic included
    % samples
    prob_rnd_rescaled = prob_rnd / (1 - psum_det);
    wgts_rnd = 1./sqrt(s_rnd * prob_rnd_rescaled);
    
    % Combine repeats and scale by sqrt(cnt)
    if combrep
        [midx_rnd,IA,mm] = unique(midx_rnd,'row');
        midx_cnts = accumarray(mm,1);
        wgts_rnd = wgts_rnd(IA).*sqrt(midx_cnts);
    end
    
    % Concatenate the deterministic indices
    wgts = vertcat(ones(s_det,1), wgts_rnd);
    midx_det = tt_ind2sub64(n(mrng), lidx_det);
    midx = vertcat(midx_det, midx_rnd);       

end
timings.sampling = toc(ss_timer);

%% Extract from KRP
% Zs = matrix of size S x R with 
% S = size(MIDX,1) and R = size(U{mrng(1)},2)

ss_timer = tic;
Zs_unweighted = extract_krp_fibers(A(mrng), midx);
timings.extract_krp = toc(ss_timer);

ss_timer = tic;
Zs = wgts .* Zs_unweighted; % reweight rows
timings.reweight_krp = toc(ss_timer);

%% Extract from Tensor
% Xst = 'transposed' matrix of size N(K) x S with
% N(K) = size(X,K) and S = size(MIDX,1);

ss_timer = tic;
if (isempty(Xfidx))
    % If X is dense, Xfidx is not used
    % If X is sparse, Xfidx will be calcualted by fibers
    Xks_unweighted = fibers(X,k,midx);
else
    Xks_unweighted = fibers(X,k,midx,Xfidx);
end
timings.extract_tensor = toc(ss_timer);

ss_timer = tic;
Xks = Xks_unweighted * diag(sparse(wgts)); % reweight columns
timings.reweight_tensor = toc(ss_timer);

%% Solve using QR
% Note that QR seems to be at least 5X faster than MATLAB's solve commands!
ss_timer = tic;
[QQ, RR] = qr(Zs, 0);
QtXkt = QQ' * Xks';  % Note that this seems to be faster than (Xks*QQ)'
V = transpose(RR \ QtXkt);
timings.solve = toc(ss_timer);

%% Finish timing of actual work
totaltime = toc(ss_overall_timer);

%% Compute sampled residual and other stuff
% Calc_res enumerated three possible options for the run
% 0: Only save out the values nedded for cp_arls_lev (Default)
% 1: Save out the samples chosen and their probabilities
% 2: Also cacluate the residual of the sampled system
ss_timer = tic;
if calcres >= 1
    if calcres == 2
        info.ss_residual = norm(Xks - V * Zs', 'fro');
    end
    info.midx = midx;
    info.midx_det = midx_det;
    info.pvec = prob_det;
    info.wgts = wgts;
    info.nnzsamp = nnz(Xks);
end

%% Save stuff
% Save out extra parameters
info.sachieved = size(midx, 1);
info.sdet = s_det;
info.pdet = psum_det;
info.totaltime = totaltime;
info.timings = timings;
info.alpha = alpha;
info.posttime = toc(ss_timer);
end


% Sub-functions
function [lidx_det, sdet, pdet, pvec] = find_deterministic_krp(lev, tau, s)

%FIND_DETERMINISTIC_KRP Generate sample indces from KRP via leverage scores.
%
%   [LIDX_DET, SDET, PDET, PVEC] = FIND_DETERMINISTIC_KRP(LEV,TAU, S) returns
%   the linear index LIDX_DET of all indices in the Khatri-Rao product for
%   which the sampling probability is greater than the threshold TAU.  It
%   also returns the number of such indices SDET, the total probability
%   assigned to these indices PDET and the associated vector of
%   probabilities PVEC.  This output can then be used to deterministically
%   include these indices when sampling by the leverage scores.  If more
%   than S indices are above the threshold, only the top S indices by
%   probability are returned.

% Brett Larsen, Tammy Kolda, 2020

d = length(lev);
n = cellfun(@(x) size(x,1),lev);

% Make sorted leverage scores
lev_top = ones(1, d);

% Important that r is a row vector for passing to the ind2sub function
r = ones(1, d);


for k = 1:d
    lev_top(k) = max(lev{k}./ sum(lev{k}));
end

% It is important here that we have left a 1 in the mode not being used
lev_top_prod = prod(lev_top);


%% Calculate the cutoff for each mode
top_prob = cell(d, 1);
top_idx = cell(d, 1);

for k = 1:d
    thresh_current = tau/lev_top_prod * lev_top(k);
    
    U_current = lev{k} ./ sum(lev{k});
    U_current(U_current < thresh_current) = 0;
    
    [top_idx{k},~, top_prob{k}] = find(U_current);
    r(k) = length(top_prob{k});

end

%% Extract the relevant indices for the top indexes
prob_temp = khatrirao({top_prob{1:d}}, 'r');

lidx_krp = find(prob_temp > tau);
pvec = prob_temp(lidx_krp);
pdet = sum(pvec);

%% Extract the indices of prob_deterministic
idx_aboveThresh = tt_ind2sub64(r, uint64(lidx_krp));
sdet = length(lidx_krp);

if (isempty(idx_aboveThresh))
    midx_det = [];
    lidx_det = [];
else 
    
    midx_det = zeros(sdet, d);
    for k = 1:d
        midx_det(:,k) = top_idx{k}(idx_aboveThresh(:, k));
    end
    
    % Transpose on n is to make it a row vector
    lidx_det = tt_sub2ind64(n.', midx_det);
    
end

%% If sdet > s, only return top s indices

if sdet > s
    [pvec_sort, idx_sort] = sort(pvec, 'descend');
    pdet = sum(pvec_sort(1:s));
    pvec = pvec_sort(1:s);
    lidx_det = lidx_det(idx_sort(1:s));
    sdet = s;
    fprintf('WARNING: Using only determinsitic samples')
end

end

function [midx, prob, alpha] = sample_krp_leverage(lev, s, c)
%SAMPLE_KRP_LEVERAGE Generate sample indces from KRP via leverage scores.
%
%   [MIDX,PROB] = SAMPLE_KRP_LEVERAGE(LEV,S,C) samples S samples according
%   to the leverage scores in the d-way cell array LEV. Each sample is a
%   d-tuple that is sampled in a way that is roughly proportional to the
%   leverage scores but damped so that the maximum number of expected
%   copies is C. The function returns the sampled multi-indices (MIDX)
%   along with the corresponding probabilities in PROB. MIDX(I,:) is a
%   d-tuple contained in the I-th sample. PROB(I) is the damped probability  
%   of sampling item I. The sampling is with replacement, meaning that the 
%   same index tuple may appear multiple times in MIDX. 

% Brett Larsen, Tammy Kolda, 2020

%% Set up
d = length(lev);
n = cellfun(@(x) size(x,1),lev);

% Rescale the leverage scores to be probabilities
p = cell(d,1);
for k = 1:d
    p{k} = lev{k} ./ sum(lev{k});
end

%% Compute damped probabilities if c is not empty

if (isempty(c))
    pdamped = p;
    alpha = [];
else
    % Extract max probability in each mode
    pmax = cellfun(@max, p);

    % Compute damping factors where
    %  * n = d-array of sizes
    %  * s = number of samples
    %  * c = max expected number of copies of any sample
    alpha = damping_factors(pmax,n,c/s);

    % Compute damped probabilities
    pdamped = cell(d,1);
    for k = 1:d
        pdamped{k} = alpha(k) * p{k} + (1-alpha(k)) * 1/n(k);
    end
end

%% Create samples

% Get indices using weighted sampling with replacement, getting all the
% samples for each mode at once
midx = zeros(s,d);
modeprobs = zeros(s,d);
for k = 1:d
    % Sample indices with replacement for weights in pdamped{k}
    midx(:,k) = tt_random_sample(pdamped{k}, s);
    % Extract probabilities of sample
    modeprobs(:,k) = pdamped{k}(midx(:,k));
end

% Compute final probability for the weights
prob = prod(modeprobs,2);
end

function Z = extract_krp_fibers(A,midx)
%EXTRACT_KRP_FIBERS Extract sampled indices from KRP.
%
%   Z = EXTRACT_KRP_FIBERS(A,MIDX) samples the specified indices from a
%   Khatri-Rao product without actually forming it explicitly. 
%
%   - A = cell array of D matrices with same number of columns
%   - A{K} = matrix of size N(K) x R for K = 1,...,D
%   - MIDX = S x D array of sampled multi-indices
%   - MIDX(I,:) = multi-index corresponding to sample I for I = 1,...,S 
%   - MIDX(I,K) = integer in the range {1,...,N(K)}   
%
%   The returned matrix Z is of size S x R where 
%   S = SIZE(MIDX,1) and R = SIZE(A{1},2).
%
%   See also FIBERS.

% Adapted from work by Casey Battaglino, circa 2016

%% Error checks on matrices 
ndimsA = cellfun(@ndims, A);
if(~all(ndimsA == 2))
    error('Each argument must be a matrix');
end

ncols = cellfun(@(x) size(x, 2), A);
if(~all(ncols == ncols(1)))
    error('All matrices must have the same number of columns.');
end

%% Calculate result
d = length(A);
matorder = d:-1:1;
Z = A{matorder(1)}(midx(:,matorder(1)),:);
for i = matorder(2:end)
    Z = bsxfun(@times, Z, A{i}(midx(:,i),:));
end
end

function alpha = damping_factors(q,n,ub)
%DAMPING_FACTORS Figure out which modes to damp and how much.
%
%   ALPHA = DAMPING_FACTORS(Q,N,UB) computes the amount of damping to get
%   the product of maximum probabilities from several distributions to be
%   equal to UB. The original maximum probabilities are given by Q and the
%   corresponding lengths are given by N. The damped distribution is given
%   by 
%
%      PNEW{K} = ALPHA(k) * P{K} + (1 - ALPHA(K)) * (1./N(K));
%
%   where Q(K) = max(P{K}) and N(K) = length(P{K}). The new distribution
%   will have the property that the maximum probability is exactly equal to
%   UB unless prod(Q) <= UB already. It should always be the case that each
%   ALPHA(K) is in the range [0,1].

%   Note that this function works RECURSIVELY. This was the easiest way to
%   track which modes corresponded to special cases.
%
% Tammy Kolda, April 2020

% Get d
d = length(q);

if ub >= 1
    alpha = ones(size(q));
    return;
end

% Set mode-wise target. If we can, we just have the maximum probability be
% the same in every mode. There are two cases where that doesn't work,
% which we check for below.
gamma = nthroot(ub,d);

% Create alpha array
alpha = -1 * ones(size(q));

% Case 1: Check for q-values that are already smaller than the target.
% These modes will not be damped. Instead, the other modes will be damped
% even more by adjusting the upper bound and recursing.
small_q = (q <= gamma);
if any(small_q)
    % Remove the modes from damping and recurse on remainder
    big_q = ~small_q;
    new_ub = ub / prod(q(small_q));
    alpha_big = damping_factors(q(big_q), n(big_q), new_ub);
    alpha(small_q) = 1;
    alpha(big_q) = alpha_big;
    return;
end

% Case 2: Check for any small modes so that uniform sampling cannot get as
% small as the target. In this case, we have to take what we can get from
% the uniform sampling and the redistribute the rest of the weight to the
% remaining modes and recurse.
small_n = (1./n > gamma);
if any(small_n)
    % Remove small modes and recurse on remainder
    big_n = ~small_n;
    new_ub = ub * prod(n(small_n));
    alpha_big = damping_factors(q(big_n), n(big_n), new_ub);
    alpha(small_n) = 0;
    alpha(big_n) = alpha_big;    
    return;     
end

% No gotchas. Can be handled in the straightforward way.
alpha = (gamma - 1./n) ./ (q - 1./n);
end
