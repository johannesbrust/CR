%------------------------- poly_train ------------------------------------%
%
% Main training loop for training a polynomial model to a sin functions
% dataset.
%
% This loop enables the choice of different functions to compute
% a search direction
%
%-------------------------------------------------------------------------%
%01/20/24, J.B., Initial implementation
%02/29/24, J.B., Update to use compact representation
%03/01/24, J.B., Updates to also print "prediction accuracy"
%03/12/24, J.B., Implementation of polynomial fitting

% Initializations
clc;
clear;
close all;
addpath(genpath('./Solvers/'))
addpath(genpath('../../SOLVER'))

%
% Initialize random seed
%
rng(0);

%
% Loop parameters and data
%
mnist       = load('mnist.mat');
[n1,n2,N]   = size(mnist.training.images);  % data size
datidx      = randperm(N);                  % reshuffled indices

N           = 2000;
NT          = 500;
x           = (linspace(-pi,pi,N))';
y           = sin(x);

x_test      = (linspace(pi,2*pi,NT))';
y_test      = sin(x_test);

btch        = N;                            % batch size 256, N
nep         = 10;                         % number of epochs, 20, 1
pltf        = 1;                            % plot loss over epochs
pltit       = 1;                            % plot iteration details
nbtch       = ceil(N/btch);                 % number of batch blocks
sol         = 'compact'; % sgd, adam, compact                      % solver to run
FF          = [];                           % function values to store
AA          = [];

%NT          = size(mnist.test.images,3);

batchfac    = 1.0; % 1.2, 0.7, 1, 0.8 1.3;
bts         = nbtch;                        % shrink/grow fact. for batch
xbtch       = N;                            % maximal batch size
mbtch       = 256;                          % minimal batch size

%
% Parameter initialization
%
nx          = n1*n2;                        % Size of image pixels

d           = 4;                        % Size of parameters

w           = 0*randn(d,1);                % Initial parameters

W           = zeros(d,3);                  % Buffer of additional optim pars.

initStrat   = 0;                            % Initialization strategy

%
% Print outputs
%
if pltit == 1
    fprintf('#########################################################\n');
    fprintf('#\n');
    fprintf('# Polynomial fitting \n');
    fprintf('# \n')
    fprintf('# N    = %i \t (data size) \n', N);
    fprintf('# Btch = %i \t (batch size) \n',btch);
    fprintf('# d    = %i \t (num. vars) \n',d);    
    fprintf('#\n');
    fprintf('# Alg  = %s \t (algorithm) \n',sol);
    fprintf('# \n');
    fprintf('#########################################################\n');
    newline;
    newline;
    head = 'Epoch   \t Iter    \t Loss    \t Accr    \t norm(w) \t norm(g) \n';
    fprintf(head);
end

if pltf == 1
    % Print the loss over epochs
    % figure;
    % title(sol);
    % xlabel('$k$','Interpreter','latex','FontSize',12);
    % ylabel('$\textnormal{Loss}$','Interpreter','latex','FontSize',12);
    % 
    % pause;
end

% Compute initial accuracy
%

aa = 0;
for ii=1:NT
    aa = polyobji(w,x_test,y_test);
end

aa  = aa;
AA  = [AA;aa]; %#ok<AGROW>

%
% Implementation of the main loop
%
for k = 1:nep

    

    for i = 1:nbtch

        %
        % Compute indices for a data pair, and the objective loss
        % and gradient
        %
        [si,ei] = btchidx(i,btch,N);

        % [fw,gw] = logobj(w,mnist.training.images(:,:,si:ei),...
        %     mnist.training.labels(si:ei));

        [fw,gw] = polyobji(w,x(si:ei),y(si:ei));

        %display(gw);

        FF = [FF;fw];                           % Storing function values   

        
        %
        % Call an optimization algorithm
        %
        switch(sol)
            case 'sgd'
                pars    = sgdpars();
                alp     = pars.alp;

                w = sgd(w,gw,[],alp);
            case 'adam'
                pars    = adampars();
                alp     = pars.alp;
                bet1    = pars.bet1;
                bet2    = pars.bet2;
                kadm    = (k-1)*nbtch + i;
                epsadm  = pars.eps;

                [w,W] = adam(w,gw,W,alp,bet1,bet2,kadm,epsadm);
            case 'compact'

                % func = @(w)(logobj(w,mnist.training.images(:,:,si:ei),...
                %                 mnist.training.labels(si:ei)));

                func = @(w)(polyobji(w,x(si:ei),y(si:ei)));

                state.it = k*i;
                state.fk = fw;
                state.gk = gw;

                opts.maxIt = 1; % 1
                opts.print = 1;

                %opts.whichV = 'g';

                %opts.whichL = 'fixed';

                opts.alp = 1;
                opts.whichL = 'fixed';
                
                opts.whichV = 's'; % s, g, ag, y

                opts.ispd = 0;

                opts.nrmtol = 'inf';

                if initStrat == 1
                    if k < 3 % 10
                        opts.whichV = 's';
                    else
                        %sol = 'adam';
                         opts.whichV = 'g';
                        opts.whichL = 'fixed';
                        %opts.whichV = 'g';
                    end
                end
                

                % Apply the compact solver to a "deterministic" problem
                if nep == 1
                    opts.maxIt = 5; % 1
                    opts.print = 1;
                    opts.l = 100;
                    [ w,fx,outs ] = compLS1(w, func, opts );
                else
                    opts.l = 10; % 5, 50
                    [w,~,~,~,state] = compLS1Step(w, func, opts, state );
                end
                %display(state.lidx);

        end

    end

    %
    % Compute current model accuracy
    %
    aa = 0;
    for ii=1:NT
        aa = polyobji(w,x_test,y_test);
    end
    
    aa  = aa;
    AA  = [AA;aa]; %#ok<AGROW>
    
    %
    % Print current results
    %
    if pltit == 1
        fprintf('%i       \t %i     \t %0.3g   \t %0.3g   \t  %0.3g   \t  %0.3g   \t \n',...
            k,nbtch*k,fw,aa,norm(w),norm(gw));
    end

    if pltf == 1
        % Show loss
        % kk = (k-1)*nbtch+1;
        % plot(kk,fw,'b.'); hold on;
    end

    btch = min(round(batchfac*btch),xbtch);
    btch = max(mbtch,btch);
    nbtch = ceil(N/btch);

    bts = [bts;nbtch]; %#ok<AGROW>

end





