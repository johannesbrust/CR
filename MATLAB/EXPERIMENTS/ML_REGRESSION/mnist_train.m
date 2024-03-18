%------------------------ mnist_train ------------------------------------%
%
% Main training loop for training a logistic model to the MNIST
% digits dataset.
%
% This loop enables the choice of different functions to compute
% a search direction
%
%-------------------------------------------------------------------------%
%01/20/24, J.B., Initial implementation
%02/29/24, J.B., Update to use compact representation
%03/01/24, J.B., Updates to also print "prediction accuracy"
%03/18/24, J.B., Preparation for release

% Initializations
clc;
clear;
close all;
addpath(genpath('./Solvers/'))
addpath(genpath('../../SOLVER'))
addpath(genpath('../../EXTERNAL/githubfetch'))
addpath(genpath('./mnist_mat-main/'))

%
% Initialize random seed
%
rng(0);

% Load data
if exist('mnist.mat','file') ~= 2
    fprintf('Setup: #### Downloading dataset\n');
    fprintf('Setup: ########### ca. 30 secs.\n');
    unzip('https://github.com/johannesbrust/mnist_mat/archive/refs/heads/main.zip');
    fprintf('Setup: ##### Download completed\n');
    fprintf('\n');
    addpath(genpath('./mnist_mat-main/'));
end
mnist       = load('mnist.mat');

%
% Loop parameters and data
%

[n1,n2,N]   = size(mnist.training.images);  % data size
datidx      = randperm(N);                  % reshuffled indices
btch        = 256;                            % batch size 256, N
nep         = 20;                         % number of epochs, 20, 1
pltit       = 1;                            % plot iteration details
nbtch       = ceil(N/btch);                 % number of batch blocks
sol         = 'compact'; % sgd, adam, compact                      % solver to run
FF          = [];                           % function values to store
AA          = [];
NT          = size(mnist.test.images,3);

batchfac    = 1.0; % 1.2, 0.7, 1, 0.8 1.3;
bts         = nbtch;                        % shrink/grow fact. for batch
xbtch       = N;                            % maximal batch size
mbtch       = 256;                          % minimal batch size

%
% Parameter initialization
%
nx          = n1*n2;                        % Size of image pixels
d           = 10*nx;                        % Size of parameters
w           = 0*randn(d,1);                % Initial parameters
W           = zeros(d,3);                  % Buffer of additional optim pars.

% Compact representation
opts.maxIt  = 1;                            % inner iterations
opts.print  = 0;                            % print solver outputs
opts.alp    = 0.5;                          % fixed learning rate
opts.whichL = 'fixed';                      % learning strategy
opts.whichV = 'y'; % s, g, ag, y            % v strategy
opts.ispd   = 0;                            % positive definiteness
opts.l      = 1; % 5, 50                    % limited-memory size

%
% Print outputs
%
if pltit == 1
    fprintf('#########################################################\n');
    fprintf('#\n');
    fprintf('# MNIST Dataset for digit classification \n');
    fprintf('# \n')
    fprintf('# N    = %i \t (data size) \n', N);
    fprintf('# nx   = %i \t (img pixels) \n',nx);
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

% Compute initial accuracy
%

aa = 0;
for ii=1:NT
    [pv,py_] = max(predct(w,mnist.test.images(:,:,ii)));
    py       = py_-1;            
    aa       = aa +(py==mnist.test.labels(ii));
end

aa  = aa/NT;
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

        [fw,gw] = logobj(w,mnist.training.images(:,:,si:ei),...
            mnist.training.labels(si:ei));

        FF = [FF;fw];                 %#ok<AGROW> % Storing function values   

        
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

                func = @(w)(logobj(w,mnist.training.images(:,:,si:ei),...
                                mnist.training.labels(si:ei)));

                state.it = k*i;
                state.fk = fw;
                state.gk = gw;

                [w,~,~,~,state] = compLS1Step(w, func, opts, state );
               
        end

    end

    %
    % Compute current model accuracy
    %
    aa = 0;
    for ii=1:NT
        [pv,py_] = max(predct(w,mnist.test.images(:,:,ii)));
        py       = py_-1;            
        aa       = aa +(py==mnist.test.labels(ii));
    end

    aa  = aa/NT;
    AA  = [AA;aa]; %#ok<AGROW>

    %
    % Print current results
    %
    if pltit == 1
        fprintf('%i       \t %i     \t %0.3g   \t %0.3g   \t  %0.3g   \t  %0.3g   \t \n',...
            k,nbtch*k,fw,aa,norm(w),norm(gw));
    end

    btch = min(round(batchfac*btch),xbtch);
    btch = max(mbtch,btch);
    nbtch = ceil(N/btch);

    bts = [bts;nbtch]; %#ok<AGROW>

end

save([sol,'.mat'],'FF','AA','bts');






