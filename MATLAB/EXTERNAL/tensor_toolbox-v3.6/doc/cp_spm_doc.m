%% Subspace Power Method for CP Decomposition of Symmetric Tensors
%
% <html>
% <p class="navigate">
% &#62;&#62; <a href="index.html">Tensor Toolbox</a> 
% &#62;&#62; <a href="cp.html">CP Decompositions</a> 
% &#62;&#62; <a href="cp_spm_doc.html">CP-SPM</a>
% </p>
% </html>
%
% The function |cp_spm| computes the symmetric CP decomposition of a 
% symmetric tensor using the Subspace Power Method (SPM). The symmetric CP
% decomposition is described in <cp_sym_doc.html Symmetric CP Decomposition>,
% while SPM is described in the following reference:
%
% * J. Kileel, J. M. Pereira, Subspace power method for symmetric tensor
%   decomposition and generalized PCA, 
%   <https://arxiv.org/abs/1912.04007 arXiv:1912.04007>, 2020

%% Create a sample problem 
% For consistency, we use the same example as in <cp_sym_doc.html Symmetric CP 
% Decomposition>
d = 3; % order
n = 10; % size
r = 2; % true rank

rng(5); % Set random number generator state for consistent results

info = create_problem('Size', n*ones(d,1), 'Num_Factors', r, ...
    'Symmetric', 1:d, 'Factor_Generator', @rand, 'Lambda_Generator', @rand, 'Noise', 0.1);

X = info.Data;
M_true = info.Soln; 
S_true = symktensor(M_true); % Convert from ktensor to symktensor

%%
% Check that the tensor is symmetric
issymmetric(X)

%% Run CP-SPM 
% SPM estimates the rank by picking a cut-off of the eigenvalues of the 
% tensor flattening. We plot these eigenvalues as follows:
cp_spm(X, 'rank_sel', 'plot_only');

%%
% We observe the first two eigenvalues contain most of the energy, and 
% conclude the tensor is approximately rank 2

rng(5); % Set random number generator state for consistent results

tic
[S, info] = cp_spm(X, 2);
toc

fprintf('\n');
fprintf('Final function value: %.2g\n', fg_explicit(S, X, norm(X)^2));
fprintf('Check similarity score (1=perfect): %.2f\n', score(S, S_true));
fprintf('\n');

%% Compare with CP-SYM using L-BFGS from Poblano Toolbox
% We compare SPM with |cp_sym|. Its options are explained in <cp_sym_doc.html 
% Symmetric CP Decomposition>; This is the recommended way to run the 
% method:
optparams = lbfgs('defaults'); % Get the optimization parameters
optparams.RelFuncTol = 1e-10; % Tighten the stopping tolerance
optparams.StopTol = 1e-6; % Tighten the stopping tolerance
rng(5); % Set random number generator state for consistent results

tic
[S,info] = cp_sym(X, 2,'unique',false,'l1param',0,'alg_options',optparams);
toc

fprintf('\n');
fprintf('Final function value: %.2g\n', fg_explicit(S, X, norm(X)^2));
fprintf('Stopping condition: %s\n', info.optout.ExitDescription);
fprintf('Check similarity score (1=perfect): %.2f\n', score(S, S_true));
fprintf('\n');