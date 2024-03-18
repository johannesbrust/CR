%------------------------ eigenComp --------------------------------------%
%
% Comparison of Eigenvalue computations
% Data from the compact representation on the Rosenbrock function
% is used.
%
% The approaches are:
%
% Direct 
% Compact QR
% Compact QR precomputed
%
%-------------------------------------------------------------------------%
% 02/12/24, J.B., 
% 02/23/24, J.B.,
% 03/18/24, J.B., preparation for release

clc;
clear;

% Add relevant paths
datapath = '../DATA/';
addpath(genpath(datapath));

%
% Settings for loading data
%
nsend = 13; % 10, 13
ns    = 2.^(3:nsend);
ksave = 10;
m     = 5;

%
% Containers to store results for comparison 
%
lns     = length(ns);
times   = zeros(lns,3);
errs    = zeros(lns,2);

fprintf('Compact Representation (EIG)  \n');
fprintf('\n');
fprintf('num. probs =  %i          \n',lns);
fprintf('smallest n =  %i          \n',2^3);
fprintf('largest n  =  %i          \n',2^nsend);
fprintf('memory     =  %i          \n',m);
fprintf('--------------------------------\n');
fprintf('n       \t Eig      \t Comp. qr     \t Comp. qr (pre)    \t err        \t err (pre)       \n');

%
% Initialize Storage containers
%
VV = zeros(m,m);
VS = zeros(m,m);
VY = zeros(m,m);
SS = zeros(m,m);
SY = zeros(m,m);
M  = zeros(2*m,2*m);
N  = zeros(2*m,2*m); 
Zm = zeros(m,m);
%VV = zeros(m,m);

%
% Parameters for eigenfactorization
%
thrs = 1.0e-11;     % Threshold for numerically linear dependence


% Start loop loading data and performing the factorizations
for i=1:lns
    n       = ns(i);
    data    = load(['comp_n_',num2str(n),'_k_',num2str(ksave),'.mat']);
    Vk      = data.Vk;
    Sk      = data.Sk;
    Yk      = data.Yk;
    R1      = data.R1;
    R2      = data.R2;
    M2      = data.M2;
    dk      = data.dk;
    gam     = data.gam;
    ll      = data.ll;

    %display(ll);

    %
    % Computation of components for the various approaches
    %
    idx     = 1:ll;
    idx2    = [idx m+idx];
    midx    = 1:2*ll; 
    %midxT   = midx; 
    M(idx2,idx2)  = [Zm(idx,idx), R2(idx,idx); R2(idx,idx)',...
                    R1(idx,idx)+R1(idx,idx)'-(diag(dk(idx))+...
                    gam.*M2(idx,idx))];       
    U           = [Vk(:,idx), Sk(:,idx)-gam.*Yk(:,idx)];
    VV(idx,idx) = Vk(:,idx)'*Vk(:,idx);
    VS(idx,idx) = Vk(:,idx)'*Sk(:,idx);
    VY(idx,idx) = Vk(:,idx)'*Yk(:,idx);
    SS(idx,idx) = Sk(:,idx)'*Sk(:,idx);
    SY(idx,idx) = Sk(:,idx)'*Yk(:,idx);
    N(idx2,idx2)  = [VV(idx,idx),(VS(idx,idx)-gam.*VY(idx,idx));...
           (VS(idx,idx)-gam.*VY(idx,idx))',...
           (SS(idx,idx)-gam.*(SY(idx,idx)+SY(idx,idx)')+gam^2.*M2(idx,idx))];
    UU = U'*U;

    % Direct approach    
    Hk          = gam.*eye(n) + U*(M(idx2,idx2)\U');
    ts          = tic;
    E           = eig(Hk);
    times(i,1)  = toc(ts);

    % Compact QR
    ts      = tic;
    RC      = qr(U,'econ');
    midxT   = midx(abs(diag(RC))>thrs);
    ec      = eig(RC(midxT,:)*(M(idx2,idx2)\(RC(midxT,:)')));
    EC      = gam + [ec; zeros(n-length(ec),1) ];
    times(i,2) = toc(ts);

    % Compact QR (precomputed)
    ts      = tic;
    %[L,D,p] = ldl(N(idx2,idx2),'vector');
    [L,D,p] = ldl(UU,'vector');
    D       = abs(D);
    midxT   = midx(abs(diag(D))>thrs);
    
    ip(p)   = 1:length(p); %#ok<SAGROW>
    
    ecp     = eig(((L(ip,midxT)*sqrt(D(midxT,midxT)))')*(M(idx2,idx2)\((L(ip,midxT)*sqrt(D(midxT,midxT))))));
    ECP     = gam + [ecp; zeros(n-length(ecp),1) ];
    times(i,2) = toc(ts);

    %display([sort(E) sort(EC) sort(ECP)]);
    
    errs(i,1) = norm(sort(E,'ComparisonMethod','real')-sort(EC,'ComparisonMethod','real'));
    errs(i,2) = norm(sort(E,'ComparisonMethod','real')-sort(ECP,'ComparisonMethod','real'));

    % Save the computed eigenvalues for the first two problems
    if i <= lns
        save([datapath,'eigs_n_',num2str(n)],'E','EC','ECP');
    end

    % Print
    fprintf([' %i      \t %0.2g      \t %0.2g      \t %0.2g           \t',...
        '%0.3e       \t %0.3e       \t \n'],n,times(i,1),times(i,2),...
         times(i,3),errs(i,1),errs(i,2));
end

% Store time and errors
save([datapath,'timesErrs_np_',num2str(lns)],'times','errs');



















