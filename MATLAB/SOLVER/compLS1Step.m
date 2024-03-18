function [ x, fx, gk, outs, state ] = compLS1Step(x, func, opts, state )
%compLS1Step Implementation of a limited-memory line-search method
% to use in a machiene learning loop
%
% This function either calls a strong-Wolfe or an armijo line-search
%
% INPUTS:
%   x: Initial solution estimate 
%   func: Function "f" and gradient "g". 
%       This function is called [f,g]=func(x);
%   opts (Structure with optional fields)
%       opts.tol: tolerance to determine a "minimum"        
%       opts.maxIt: Maximum iterations
%       opts.print: Print outputs
%       opts.whichL: Switch for different line-search
%       opts.whichV: Switch for different V strategies
%       opts.l: Limited-memory paramter
%       opts.store: Storing function and gradient traces
%       ------ Line-search paramters ------
%       opts.step0: Initial step length to try
%       opts.c1: Armijo condition (sufficient decrease)
%       opts.c2: Curvature condition 
%       opts.stepMax: Max. step
%       opts.stepMin: Min. step
%       opts.maxItLS: Max. iterations for line-search
%       state: input information for the compact representation
% OUTPUTS:
%   x: Approximation of minimum
%   fx: Function value at minimum
%   outs: output struct
%   state: updated state information
%
%--------------------------------------------------------------------------
% 02/05/24, J.B.
% 02/11/24, J.B., Implementation of compact representation
% 02/12/24, J.B., Addition of a storage flag
% 02/21/24, J.B., Interface change
% 02/21/24, J.B., Enabling other norms for convergence
% 02/26/24, J.B., Bug fix, for limited memory updates
%                 and implementation of a restart strategy. Saving traces
%                 of function and gradient values.
% 02/29/24, J.B., Implementation of the step version for a ML loop
% 03/01/24, J.B., Implementation of a fixed stepsize method
% 03/05/24, J.B., Modification to the fixed stepsize method
%                 Option to suppress pd requirement

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
    whichL = 'wolfeG';
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
if isfield(opts,'nrmtol')
    nrmtol = opts.nrmtol;
else
    nrmtol = '2';
end
if isfield(opts,'store')
    store = opts.store;
else
    store = false;
end
if isfield(opts,'alp')
    alp = opts.alp;
else
    alp = 0.01;
end
if isfield(opts,'ispd')
    ispd = opts.ispd;
else
    ispd = 1;
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

% State initializations

it = state.it;
fk = state.fk;
gk = state.gk;
ng = norm(gk);

if it==1    
    p = -gk/ng;
    % Initializations
    Vk = zeros(n,l);
    Sk = zeros(n,l);
    Yk = zeros(n,l);
    R1 = zeros(l,l); % triu(Sk'Yk)
    R2 = zeros(l,l); % triu(Vk'Yk)
    M2 = zeros(l,l); % Yk'Yk
    dk = zeros(l,1); % diag(Sk'Yk), this is a vector l x one    
    Wk = zeros(n,l);
    lidx = 1:l;      % limited-memory indices    
    ll   = 0;
    gam  = 1.0;
else
    p  = state.p;
    Vk = state.Vk;
    Sk = state.Sk;
    Yk = state.Yk;
    R1 = state.R1;
    R2 = state.R2;
    M2 = state.M2;
    dk = state.dk;
    Wk = state.Wk;
    lidx = state.lidx;
    ll   = state.ll;
    gam  = state.gam;
end

% Start timing
ts = tic;

sk = zeros(n,1);
yk = zeros(n,1);
vk = zeros(n,1);
b1 = zeros(l,1); % buffer 1
b2 = zeros(l,1); % buffer 2
b3 = zeros(l,1); % buffer 3

v1   = zeros(n,1); % Memory for computing average values
bet1 = 0.9; % 0.9 



infs    = '   ';

LSUT.UT = true; LSUTT.UT = true; LSUTT.TRANSA = true;   % linsolve opts.

dxMax       = 100;
dxInf       = 1.0e+10;
updateTol   = 0.99;

% fk  = func(x);
% gk  = grad(x);
% 
% [fk,gk] = func(x);
% ng      = norm(gk);

if strcmp(nrmtol,'2'); ngtol = ng; else; ngtol = norm(gk,nrmtol); end

k   = 0;
nf  = 0;        % number of function evaluations
nskp= 0;        % number of QN skips
nsd = 0;        % number of steepest descent steps
step= 1.0;
ex  = 0;        % exit flag
%ll  = 0; % Amount of limited-memory stored 

%p   = - gk/ng;

if printB==true        
   fprintf('%i \t %i \t %0.5e      \t %0.5e       \t %0.3e     \t %i\n',...
       k,nf,fk,ng,step,0);
end

% Storing iteration information
if store
    fs = zeros(maxIt,1);
    gs = zeros(maxIt,1);
    fs(k+1) = fk;
    gs(k+1) = ngtol;
else
    fs = 0;
    gs = 0;
end

% % Initial line-search to generate s0,y0,v0
% [step,x1,f1,g1,jf,iExit] = wolfeG(func,grad,wolfeTols,step0,stepMax,f,g,p,x);
% nf = nf + jf;

% Main loop
while k<maxIt && ngtol > tol
    
    %p = -Hk*gk;    
    %display(gk');

    gp        = gk'*p;

    % Safeguard to use the normalized steepest descent direction
    % if the quasi-Newton step is not descent
    if gp > 0 && ispd == 1
        p = -gk/ng; 
        gp = - ng; nsd = nsd + 1; infs(1)='a'; 

        % Restarting
        ll   = 0;
        lidx = 1:l;
        infs(3) = 'R';

    end

    normp     = norm(p);
    stepMax   = dxInf/(1+normp);
    stepLimit = dxMax/(1+normp);
    step      = min([1 stepLimit stepMax]);
    
    switch whichL
        case 'armijoG'
            [step,x1,f1,g1,jf,iExit] = armijoG1(func,step,stepMax,fk,gk,p,x);
            nf = nf + jf;
        case 'wolfeG'
            wolfeTols = [c2, c1];
            [step,x1,f1,g1,jf,iExit] = wolfeG1(func,wolfeTols,step,stepMax,fk,gk,p,x);
            nf = nf + jf;
        case 'wolfeB'
            [step,x1,f1,g1,jf,iExit] = wolfeB1(x,p,gk,fk,step,c1,c2,func,maxItLS,...
                                        stepMin,2,stepMax);
            % [step,jf,jf,iExit] = wolfeB__(x,p,gk,fk,step,c1,c2,func,grad,maxItLS,...
            %                             stepMin,2,stepMax);
            nf = nf+jf;            
        case 'wolfeMT'
            [x1,f1,g1,step,iExit,jf] = cvsrch_INTF1(func,n,x,fk,gk,p,step,...
                c1,c2,1e-12,1e-12,1000,200);
            nf = nf+jf;
        case 'fixed'
            if infs(1) == 'a' || ll == 0; p = -gk; end
            
            x1      = x + alp*p;
            sk = alp*p;
            if ll==0
                alp1 = min(1,1/sum(abs(gk)));
                x1 = x + alp1*p;
                sk = alp1*p;
            end
            
            [f1,g1] = func(x1);
            nf      = nf + 1;
            iExit = 1;

            if abs(f1) == inf
                [f1,g1] = func(x - alp*gk);
                nf = nf + 1;
            end
    end

    if ~strcmp(whichL,'fixed')
        sk = x1-x;
    end
    yk = g1 - gk;
    sy = sk'*yk;

    ny  = norm(yk);
    ns  = norm(sk);

    %display(sk');
    %display(yk');

    switch whichV
        case 's'
            vk = sk;
        % case 'sn'                 % Method is scale invariant to s
        %     vk = sk/ns;
        case 'g'
            vk = -gk;          
        case 'ag'
            
            %v1 = v1 - gk;

            %v1 = bet1*v1 + (1-bet1)*(-gk);
            
            %vk = v1/max(k,1);

            vk = bet1*Wk(:,1) + (1-bet1)*(-gk);

            Wk(:,1) = vk;

        case 'y'
            vk = yk;
    end

    %updateTol1 = 1e-8;

    % Limited-memory updating
    if sy > - updateTol*(1 - c2)*step*gp || ispd == 0 % (1 - c2)*        sy > ns*ny*updateTol1 %
        
        gam = sy/ny^2;

        %gam = max(gam,1);

        if ll < l

            ll = ll + 1;
            
        else

            b1(1)             = lidx(1);
            lidx(1:ll-1)      = lidx(2:ll);
            lidx(ll)          = b1(1);
            R1(1:ll-1,1:ll-1) = R1(2:ll,2:ll);
            R2(1:ll-1,1:ll-1) = R2(2:ll,2:ll);
            dk(1:ll-1)        = dk(2:ll);
            M2(1:ll-1,1:ll-1) = M2(2:ll,2:ll);

        end

        Sk(:,lidx(ll))  = sk;
        Yk(:,lidx(ll))  = yk;
        Vk(:,lidx(ll))  = vk;        
        R1(1:ll,ll)     = Sk(:,lidx(1:ll))'*yk;
        R2(1:ll,ll)     = Vk(:,lidx(1:ll))'*yk;
        dk(ll)          = sy;
        M2(1:ll,ll)     = Yk(:,lidx(1:ll))'*yk;
        M2(ll,1:ll)     = M2(1:ll,ll);

        % Safeguard
        if sum(isnan(diag(R2(1:ll,1:ll))))>0
            idxnan = isnan(diag(R2(1:ll,1:ll)));
            display(Vk(:,lidx(1:ll))'*yk);
            display(Vk(:,lidx(1:ll))'*Vk(:,lidx(1:ll)));
            display(yk'*yk);
            display(ll);
            display(f1);
            display(norm(g1));
        end

    else
        gam = 1/norm(gk);
        nskp = nskp + 1;
        infs(2) = 's';
    end

    if strcmp(whichL,'fixed'); gam = 1.0; end

    fk = f1;
    x  = x1;
    gk = g1;

    ng = norm(gk);

    % Determining the norm
    if strcmp(nrmtol,'2'); ngtol = ng; else; ngtol = norm(gk,nrmtol); end

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
    
    % b1(1:ll) = (Vk(:,lidx(1:ll))'*yk);
    % b2(1:ll) = (Sk(:,lidx(1:ll))'*yk) - gam.* (Yk(:,lidx(1:ll))'*yk);    
    % b1(1:ll) = linsolve(R2(1:ll,1:ll),b1(1:ll),LSUT);
    % b3(1:ll) = b1(1:ll);
    % b1(1:ll) = R1(1:ll,1:ll)*b1(1:ll) + R1(1:ll,1:ll)'*b1(1:ll) - ...
    %     dk(1:ll).*b1(1:ll) - gam.*(M2(1:ll,1:ll)*b1(1:ll));
    % b1(1:ll) = linsolve(R2(1:ll,1:ll),(b2(1:ll)-b1(1:ll)),LSUTT);
    % 
    % b2(1:ll) = b3(1:ll);
    % 
    % s1 = gam.*yk + Vk(:,lidx(1:ll))*b1(1:ll) + Sk(:,lidx(1:ll))*b2(1:ll) - ...
    %         Yk(:,lidx(1:ll))*(gam.*b2(1:ll));
    % 
    % % display(sy);  
    % % display(p'*gk);  
    % % display(norm(sk));
    % % display(norm(s1));
    % display(norm(s1-sk));
    % %

    % Further testing: Compute the step via the two-loop recursion
    % if ll > 0
    % 
    %     %display(lidx(1:ll));
    % 
    %     pt = twoloop(gam,-gk,Sk,Yk,lidx(1:ll));
    % 
    %     display(norm(pt-p));
    % 
    %     st = twoloop(gam,yk,Sk,Yk,lidx(1:ll));
    % 
    %     display(norm(st-s1))
    % 
    % 
    %     %p = pt;
    % end
    
    
    % End debug

    if store
        fs(k+1) = fk;
        gs(k+1) = ngtol;
    end

    k = k + 1;
    if printB==true      
        if mod(k,20) == 0
            fprintf('k    \t nf      \t fk         \t ||gk||         \t step        \t iexit       \n');
        end
        fprintf('%i \t %i \t %0.5e      \t %0.5e       \t %0.3e     \t %i \t %s     \n',...
            k,nf,fk,ng,step,iExit,infs(1:3));
    end
    if storeCOMP==true && k == storek
        filen = ['comp','_n_',num2str(n),'_k_',num2str(k)];
        save([datapath,filen],'n','R1','R2','M2','dk','Sk','Vk','Yk',...
            'gam','k','ll');
    end    

    infs(:) = '   ';

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
outs.fs     = fs;        % trace of function values
outs.gs     = gs;        % trace of gradient norms

% Update the state
state.p     = p; 
state.Vk    = Vk;
state.Sk = Sk;
state.Yk = Yk;
state.R1 = R1;
state.R2 = R2;
state.M2 = M2;
state.dk = dk;
state.Wk = Wk;
state.lidx = lidx;
state.ll = ll;
state.gam = gam;
state.gk = gk;




