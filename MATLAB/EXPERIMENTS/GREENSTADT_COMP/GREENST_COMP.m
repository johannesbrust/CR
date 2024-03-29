%--------------------------- GREENST_COMP --------------------------------%
%
% Script to test the equivalence of the recursive rank 2 formula 
% from Dennis and More and the compact representation when vk = yk, ie.,
% the compact representation of Greenstadt's formula.
%
%`  Hk1 = Hk + thet*(rk*vk' + vk*rk) - ((rk'*yk)*thet^2) * vk*vk',    (Gen. Rank2)
%
%   with rk = sk - Hk*yk, vk=yk and thet = 1/(vk'*yk)
%
% The compact representation is
%
%   Hk1 = H0 + [Sk H0Yk] inv[ Rk+Rk'-(Dk+Yk'H0Yk) +RYk + RYk', RYk;
%                             RYk,                              0] 
%                        [Sk H0Yk]'
%
%   where Rk = triu(Sk'*Yk) and RYk = triu(Yk'*Yk)
%
%-------------------------------------------------------------------------%
% 03/04/24, J.B., Initial implementation
% 03/18/24, J.B., preparation for release

clc;
clear;

%
% Setup problem
% 
rng(0);
n   = 8;
kk  = n;
Sk  = randn(n,kk);
Yk  = randn(n,kk);
In  = eye(n);

Rk  = triu(Sk'*Yk); 
dk  = diag(Rk);
RYk = triu(Yk'*Yk);

%
% Initialize both matrices
% and apply the recursive updates
%
H0  = In; 
%HkA = In;
HkB = In;


fprintf('################################################################\n');
fprintf('#\n');
fprintf('# Test for the equivalence of the recursive rank 2 formula from \n'); 
fprintf('# Dennis and More when vk = yk, ie., the compact representation \n');
fprintf('# of Greenstadt''s formula. \n');
fprintf('# \n');
fprintf('# er1 is the error of the secant equation for the new formula  \n');
fprintf('# and er2 is the Frobenius norm error between the two matrices \n');
fprintf('#\n');
fprintf('###############################################################\n');
newline;

fprintf('k  \t er1     \t er2     \n')

for k = 1:kk

    sk = Sk(:,k);
    yk = Yk(:,k);    

    % %
    % % Standard BFGS inverse
    % %
    % rho = 1./ (sk'*yk);
    % Vk  = (In - (rho*sk)*yk');
    % HkA = Vk*HkA*Vk' + (rho*sk)*sk';
    % 
    % erA = norm(HkA*yk-sk);

    %
    % General Rank-2 formula (with Greenstadt's vk = yk)
    %
    vk      = yk;
    thet    = 1./ (yk'*vk);
    rk      = sk - HkB*yk;
    HkB     = HkB + thet*(rk*vk' + vk*rk') - (((rk'*yk)*thet^2) * vk)*vk';

    erB_ = norm(HkB*yk-sk);

    %
    % Compact representation
    %
    HkA = H0 + [Sk(:,1:k),H0*Yk(:,1:k)]* ...
        ([Rk(1:k,1:k) + Rk(1:k,1:k)'-(diag(dk(1:k))+Yk(:,1:k)'*H0*Yk(:,1:k)) ...
        + RYk(1:k,1:k) + RYk(1:k,1:k)', RYk(1:k,1:k)'; ...
        RYk(1:k,1:k), zeros(k,k)]\[Sk(:,1:k),H0*Yk(:,1:k)]');

    erA = norm(HkA*yk-sk);
    erB = norm(HkA-HkB,'fro');

    fprintf('%i \t %0.3g \t %0.3g\n',k,erA,erB);
    %fprintf('%i \t &\\texttt{%0.3g} \t &\\texttt{%0.3g} \\\\ \n',k,erA,erB);

end

% Compute error between two matrices
%erMAT = norm(HkA-HkB,'Fro');











