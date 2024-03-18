function s = tt_random_sample(cnts, nsamples)
%TT_RANDOM_SAMPLE creates a random sample proportional to the given counts.
%
%   S = RANDOM_SAMPLE(C) choose N = round(sum(C)) samples (with
%   replacement) from {1,...,length(C)} proportional to the values in C.
%   So, if C = [2 1 1], then we might expect S (sorted) to be [ 1 1 2 3 ].
%   However, we also allow for C to be non-integral.
%
%   S = RANDOM_SAMPLE(C, NSAMP) choose NSAMP samples proportional to the
%   values in C as described above.
%
%   Adapted from:
%   Tamara G. Kolda, Ali Pinar, and others, FEASTPACK v1.1, Sandia National
%   Laboratories, SAND2013-4136W, http://www.sandia.gov/~tgkolda/feastpack/, 
%   January 2014.
%
%Tensor Toolbox for MATLAB: <a href="https://www.tensortoolbox.org">www.tensortoolbox.org</a>

if ~exist('nsamples','var')
    nsamples = round(sum(cnts));
else
    cnts = cnts .* nsamples / sum(cnts);
end

cumdist = [0; cumsum(cnts)];
bins = cumdist / cumdist(end);

testval = abs(bins(end) - 1);
if  testval > eps
    warning('Last entry of bins is not exactly 1. Diff = %e.', testval);
end

[~, s] = histc(rand(nsamples,1),bins);