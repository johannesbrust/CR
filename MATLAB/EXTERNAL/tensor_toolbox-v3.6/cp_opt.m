function [M, M0, info] = cp_opt(X,r,varargin)
%CP_OPT Fits a CP model to a tensor via optimization.
%
%   M = CP_OPT(X,R) fits an R-component CANDECOMP/PARAFAC (CP) model
%   to the tensor X. The result M is a ktensor. The function being
%   optimized is F(M) = || X - M ||^2 / || X ||^2.
%
%   [M,M0,INFO] = CP_OPT(X,R) returns the initial guess in M0 and
%   additional information in info. 
%
%   [...] = CP_OPT(X,R,'param',value,...) takes additional arguments:
%
%      'method' - Optimzation algorithm. Default:'lbfgsb'. 
%         o 'lbfgsb' (Limited-Memory Quasi-Newton with bound constraints),
%         o 'lbfgs' (Limited-Memory Quasi-Newton from Poblano Toolbox), 
%         o 'fminunc'(Quasi-Newton from MATLAB Optimization Toolbox),            
%         For optimzation algorithm choices and parameters, see
%         <a href="matlab:web(strcat('file://',fullfile(getfield(what('tensor_toolbox'),'path'),'doc','html','opt_options_doc.html')))">Documentation for Tensor Toolbox Optimization Methods</a>
%
%      'init' - Initialization for factor matrices (default: 'randn'). 
%         o a cell array with initial factor matrices,
%         o a ktensor with initial factor matrices,
%         o 'randn' (randomly generated via randn function)
%         o 'rand'  (randomly generated via rand function)
%         o 'nvecs' (leading left singular vectors of each unfolding)
%
%      'state' - Random state, to re-create the same outcome.
%
%      'scale' - The optimization can be sensitive to scaling. This option
%                changes the demoninator of the objective function, i.e.,
%                to  F(M) = ||X-M||^2 / S. If the optimization is
%                converging prematurely due to lack of improvement in the
%                function value, try setting S = ||X||^2 / C so that S is
%                less than O(1e10). The default is S = ||X||^2.
%
%   Optimization and other options are discussed in the <a href="matlab:web(strcat('file://',fullfile(getfield(what('tensor_toolbox'),'path'),'doc','html','cp_opt_doc.html')))">documentation</a>.
%
%   REFERENCE: E. Acar, D. M. Dunlavy and T. G. Kolda, A Scalable
%   Optimization Approach for Fitting Canonical Tensor Decompositions,
%   J. Chemometrics, 25(2):67-86, 2011, http://doi.org/10.1002/cem.1335.
%
%   <a href="matlab:web(strcat('file://',fullfile(getfield(what('tensor_toolbox'),'path'),'doc','html','cp_opt_doc.html')))">Documentation page for CP-OPT</a>
%
%   See also CP_ALS, CP_WOPT, GCP_OPT, TENSOR, SPTENSOR, KTENSOR.
%
%Tensor Toolbox for MATLAB: <a href="https://www.tensortoolbox.org">www.tensortoolbox.org</a>
%
%--------------------------------------------------------------------------
% 02/21/24, J.B., modification to include the compact solver
%                 Note: To use the compact solver, it must be on the search
%                 path

%% Hidden options

% 'Xnormsqr' - Value for constant term ||X||^2 in objective function.
%  If not specified, will be computed. Set to scalar value to skip.
%

%%
tic; 

%% Random set-up
defaultStream = RandStream.getGlobalStream;

%% Algorithm Parameters
params = inputParser;
params.KeepUnmatched = true;
params.PartialMatching = false;
params.addParameter('state', defaultStream.State);
params.addParameter('init','randn');
params.addParameter('method','lbfgsb');
params.addParameter('Xnormsqr', []);
params.addParameter('scale',[]);
params.addParameter('printitn',1);

params.parse(varargin{:});

%% Initialize random number generator with specified state
defaultStream.State = params.Results.state;

%% Setup
init = params.Results.init;
method = params.Results.method;
Xnormsqr = params.Results.Xnormsqr;
scale = params.Results.scale;
printitn = params.Results.printitn;

optopts = params.Unmatched;
optopts.printitn = printitn;

% Save
info.params = params.Results;
f = fieldnames(optopts);
for i = 1:length(f)
   info.params.(f{i}) = optopts.(f{i});
end
%% Initialization
sz = size(X);
d= ndims(X);

if iscell(init)
    M0 = ktensor(init);
elseif isa(init,'ktensor')
    M0 = init; 
    if ~all(M0.lambda==1)
        warning('Initial guess does not have unit weight; renormalizing.')
        M0 = normalize(M0,1);
    end
elseif strcmpi(init,'nvecs')
    U0 = cell(d,1);
    for k = 1:d
        U0{k} = nvecs(X,k,r);
    end
    M0 = ktensor(U0);
elseif strcmpi(init,'rand')
    M0 = ktensor(@rand,sz,r);
elseif strcmpi(init,'randn')
    M0 = ktensor(@randn,sz,r);
else
    error('Invalid initialization')
end
if ncomponents(M0) ~= r
    error('Initial guess has %d components but expected %d components', ncomponents(M0), r);
end

%% Compute ||X||^2
if isempty(Xnormsqr)
    Xnormsqr = norm(X)^2;
end

%% Compute scale
if isempty(scale)
    if Xnormsqr > 0
        scale = Xnormsqr;
    else
        scale = 1;
    end
end

%% Shared options
optopts.xdesc = sprintf('Size: %s, Rank: %d', tt_size2str(size(X)),r);

%% Finish setup
setuptime = toc; 

%% Optimization
tic
if (printitn > 0)
    fprintf('\nCP-OPT CP Direct Optimzation');
end
switch(method)
    case {'lbfgsb','lbfgs','fminunc','fmincon','compLS1'}
        fgh = @(x) fg(update(M0,1:d,x),X,scale,Xnormsqr,true,false,false);
        optname = sprintf('tt_opt_%s',method);
        if strcmp(method,'compLS1'); optname = method; end

        [x,f,optinfo] = feval(optname, tovec(M0,false), fgh, optopts);
    otherwise
        error('Invalid method')
end
opttime = toc;

%% Clean up
M = update(M0,1:d,x);
M = arrange(M);
M = fixsigns(M);

%% Save results

info.f = f;
info.optout = optinfo;
info.opttime = opttime;
info.setuptime = setuptime;
info.Xnormsqr = Xnormsqr;
info.scale = scale;
