function [ x,fx,outs ] = compLS(x, func, grad, opts )
%compLS Implementation of a limited-memory line-search method
%
% This function calls either a backtracking line-search algorithm
% or one of three strong-Wolfe line-searches
%
% INPUTS:
%   x: Initial solution estimate 
%   func: Function "f". This function is called f=func(x);
%   grad: Gradient "f'". This function is called g = grad(x);
%   opts (Structure with optional fields)
%       opts.tol: tolerance to determine a "minimum"        
%       opts.maxIt: Maximum iterations
%       opts.print: Print outputs
%       opts.whichL: Switch for different line-search
%       opts.whichV: Switch for different V strategies
%       opts.l: Limited-memory paramter
%       ------ Line-search paramters ------
%       opts.step0: Initial step length to try
%       opts.c1: Armijo condition (sufficient decrease)
%       opts.c2: Curvature condition 
%       opts.stepMax: Max. step
%       opts.stepMin: Min. step
%       opts.maxItLS: Max. iterations for line-search
% OUTPUTS:
%   x: Approximation of minimum
%   fx: Function value at minimum
%   outs: output struct.
%
%--------------------------------------------------------------------------
% 02/05/24, J.B.
% 02/11/24, J.B., Implementation of compact representation
% 02/12/24, J.B., Addition of a storage flag

% Initializing 
if isstruct(opts)==false;opts=struct();end
if isfield(opts,'tol')
    tol = opts.tol;
else
    tol = 1e-5;
end
if isfield(opts,'maxIt')
    maxIt = opts.maxIt;
else
    maxIt = 1000;
end
if isfield(opts,'print')
    printB = opts.print;
else
    printB = 1;
end
if isfield(opts,'whichL')
    whichL = opts.whichL;
else
    whichL = 'armijo';
end
if isfield(opts,'whichV')
    whichV = opts.whichV;
else
    whichV = 's';
end
if isfield(opts,'l')
    l = opts.l;
else
    l = 5;
end
if isfield(opts,'step0')
    step0 = opts.step0;
else
    step0 = 1.0;
end
if isfield(opts,'c1')
    c1 = opts.c1;
else
    c1 = 1e-4;
end
if isfield(opts,'c2')
    c2 = opts.c2;
else
    c2 = 0.9;
end
if isfield(opts,'stepMax')
    stepMax = opts.stepMax;
else
    stepMax = 100.0;
end
if isfield(opts,'stepMin')
    stepMin = opts.stepMin;
else
    stepMin = 1e-12;
end
if isfield(opts,'maxItLS')
    maxItLS = opts.maxItLS;
else
    maxItLS = 500;
end
if isfield(opts,'storeCOMP')
    storeCOMP   = opts.storeCOMP;
    datapath    = '../DATA/';
    storek      = 10; % Store values at iteration k=10
else
    storeCOMP = false;
end

n   = size(x,1);
if printB==true
    fprintf('Compact Representation (ALG0)  \n');
    fprintf('\n');
    fprintf('Line-search: %s          \n',whichL);
    fprintf('V-strategy:  %s          \n',whichV);
    fprintf('         n=  %i          \n',n);
    fprintf('--------------------------------\n');
    fprintf('k    \t nf      \t fk         \t ||gk||         \t step        \t iexit       \n');
end

% Start timing
ts = tic;

% Initializations
Vk = zeros(n,l);
Sk = zeros(n,l);
Yk = zeros(n,l);
R1 = zeros(l,l); % triu(Sk'Yk)
R2 = zeros(l,l); % triu(Vk'Yk)
M2 = zeros(l,l); % Yk'Yk
dk = zeros(l,1); % diag(Sk'Yk), this is a vector l x one
sk = zeros(n,1);
yk = zeros(n,1);
vk = zeros(n,1);
b1 = zeros(l,1); % buffer 1
b2 = zeros(l,1); % buffer 2
b3 = zeros(l,1); % buffer 3

lidx = 1:l;      % limited-memory indices

LSUT.UT = true; LSUTT.UT = true; LSUTT.TRANSA = true;   % linsolve opts.

dxMax       = 100;
dxInf       = 1.0e+10;
updateTol   = 0.99;

fk  = func(x);
gk  = grad(x);
ng  = norm(gk);

gam = 1.0; 

k   = 0;
nf  = 1;        % number of function evaluations
nskp= 0;        % number of QN skips
nsd = 0;        % number of steepest descent steps
step= 1.0;
ex  = 0;        % exit flag
ll  = 0; % Amount of limited-memory stored 

p   = - gk/ng;

if printB==true        
   fprintf('%i \t %i \t %0.5e      \t %0.5e       \t %0.3e     \t %i\n',...
       k,nf,fk,ng,step,0);
end

% % Initial line-search to generate s0,y0,v0
% [step,x1,f1,g1,jf,iExit] = wolfeG(func,grad,wolfeTols,step0,stepMax,f,g,p,x);
% nf = nf + jf;

% Main loop
while k<maxIt && ng > tol
    
    %p = -Hk*gk;
    
    gp        = gk'*p;

    % Safeguard to use the normalized steepest descent direction
    % if the quasi-Newton step is not descent
    if gp > 0; p = -gk/ng; gp = - ng; nsd = nsd + 1; end

    normp     = norm(p);
    stepMax   = dxInf/(1+normp);
    stepLimit = dxMax/(1+normp);
    step      = min([1 stepLimit stepMax]);
    
    switch whichL
        case 'armijoG'
            [step,x1,f1,g1,jf,iExit] = armijoG(func,grad,step,stepMax,fk,gk,p,x);
            nf = nf + jf;
        case 'wolfeG'
            wolfeTols = [c2, c1];
            [step,x1,f1,g1,jf,iExit] = wolfeG(func,grad,wolfeTols,step,stepMax,fk,gk,p,x);
            nf = nf + jf;
        case 'wolfeB'
            [step,x1,f1,g1,jf,iExit] = wolfeB(x,p,gk,fk,step,c1,c2,func,grad,maxItLS,...
                                        stepMin,2,stepMax);
            % [step,jf,jf,iExit] = wolfeB__(x,p,gk,fk,step,c1,c2,func,grad,maxItLS,...
            %                             stepMin,2,stepMax);
            nf = nf+jf;            
        case 'wolfeMT'
            [x1,f1,g1,step,iExit,jf] = cvsrch_INTF(func,grad,n,x,fk,gk,p,step,...
                c1,c2,1e-12,1e-12,1000,200);
            nf = nf+jf;
    end

    sk = x1-x;
    yk = g1 - gk;
    sy = sk'*yk;

    ny  = norm(yk);
    ns  = norm(sk);

    switch whichV
        case 's'
            vk = sk;
        % case 'sn'                 % Method is scale invariant to s
        %     vk = sk/ns;
    end

    %updateTol1 = 1e-8;

    % Limited-memory updating
    if sy > - updateTol*(1 - c2)*step*gp % (1 - c2)*        sy > ns*ny*updateTol1 %
        gam = sy/ny;

        if ll < l

            ll = ll + 1;
            
        else

            b1(1)             = lidx(1);
            lidx(1:ll-1)      = lidx(2:ll);
            lidx(ll)          = b1(1);
            R1(1:ll-1,1:ll-1) = R1(2:ll,2:ll);
            R2(1:ll-1,1:ll-1) = R2(2:ll,2:ll);
            dk(1:ll-1)        = dk(2:ll);

        end

        Sk(:,lidx(ll))  = sk;
        Yk(:,lidx(ll))  = yk;
        Vk(:,lidx(ll))  = vk;        
        R1(1:ll,ll)     = Sk(:,lidx(1:ll))'*yk;
        R2(1:ll,ll)     = Vk(:,lidx(1:ll))'*yk;
        dk(ll)          = sy;
        M2(1:ll,ll)     = Yk(:,lidx(1:ll))'*yk;
        M2(ll,1:ll)     = M2(1:ll,ll);

    else
        nskp = nskp + 1;
    end

    fk = f1;
    x  = x1;
    gk = g1;

    ng = norm(gk);

    % Search direction
    b1(1:ll) = -(Vk(:,lidx(1:ll))'*gk);
    b2(1:ll) = -(Sk(:,lidx(1:ll))'*gk) + gam.* (Yk(:,lidx(1:ll))'*gk);    
    b1(1:ll) = linsolve(R2(1:ll,1:ll),b1(1:ll),LSUT);
    b3(1:ll) = b1(1:ll);
    b1(1:ll) = R1(1:ll,1:ll)*b1(1:ll) + R1(1:ll,1:ll)'*b1(1:ll) - ...
        dk(1:ll).*b1(1:ll) - gam.*(M2(1:ll,1:ll)*b1(1:ll));
    b1(1:ll) = linsolve(R2(1:ll,1:ll),(b2(1:ll)-b1(1:ll)),LSUTT);
    
    b2(1:ll) = b3(1:ll);

    p = -gam.*gk + Vk(:,lidx(1:ll))*b1(1:ll) + Sk(:,lidx(1:ll))*b2(1:ll) - ...
            Yk(:,lidx(1:ll))*(gam.*b2(1:ll));
        
    % Testing if the compact representation satisfies the secant 
    % condition
    
%     b1(1:ll) = (Vk(:,lidx(1:ll))'*yk);
%     b2(1:ll) = (Sk(:,lidx(1:ll))'*yk) - gam.* (Yk(:,lidx(1:ll))'*yk);    
%     b1(1:ll) = linsolve(R2(1:ll,1:ll),b1(1:ll),LSUT);
%     b3(1:ll) = b1(1:ll);
%     b1(1:ll) = R1(1:ll,1:ll)*b1(1:ll) + R1(1:ll,1:ll)'*b1(1:ll) - ...
%         dk(1:ll).*b1(1:ll) - gam.*(M2(1:ll,1:ll)*b1(1:ll));
%     b1(1:ll) = linsolve(R2(1:ll,1:ll),(b2(1:ll)-b1(1:ll)),LSUTT);
%     
%     b2(1:ll) = b3(1:ll);
% 
%     s1 = gam.*yk + Vk(:,lidx(1:ll))*b1(1:ll) + Sk(:,lidx(1:ll))*b2(1:ll) - ...
%             Yk(:,lidx(1:ll))*(gam.*b2(1:ll));
%     
%     display(sy);  
%     display(p'*gk);  
%     display(norm(sk));
%     display(norm(s1));
%     display(norm(s1-sk));
    %
    % End debug

    k = k + 1;
    if printB==true      
        if mod(k,20) == 0
            fprintf('k    \t nf      \t fk         \t ||gk||         \t step        \t iexit       \n');
        end
        fprintf('%i \t %i \t %0.5e      \t %0.5e       \t %0.3e     \t %i     \n',...
            k,nf,fk,ng,step,iExit);
    end
    if storeCOMP==true && k == storek
        filen = ['comp','_n_',num2str(n),'_k_',num2str(k)];
        save([datapath,filen],'n','R1','R2','M2','dk','Sk','Vk','Yk',...
            'gam','k','ll');
    end    
end

% Check convergence
if ng < tol
    ex = 1;
elseif k==maxIt
    ex = 0;
end
fx = fk;

%Save output information
outs.time   = toc(ts);
outs.normg  = ng;
outs.it     = k;
outs.nf     = nf;        % number of function evaluations
outs.nskp   = nskp;      % number of QN skips
outs.nsd    = nsd;       % number of steepest descent steps
outs.ex     = ex;        % exit flag





