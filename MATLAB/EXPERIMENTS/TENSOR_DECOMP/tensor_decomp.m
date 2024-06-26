%------------------------ tensor_decomp ----------------------------------%
%
% tensor_decomp
%
% Computing CP tensor factorizations using either the compact 
% representation or L-BFGS-B
%
% Generate a "simulation" of problems and store the information 
% for each run
%
%-------------------------------------------------------------------------%
% 02/26/24, J.B., initial version
% 03/18/24, J.B., preparation for release

addpath(genpath('../../EXTERNAL/tensor_toolbox-v3.6/'))
addpath(genpath('../../SOLVER/'));
datapath = '../../DATA';

%
% Problem size
%
sz      = [250 250 250];            % Increase size of tensor [250 250 250]
numsim  = 15;                       % Number of "simulations"/problems


%
% Compact parameters. Adjusted to match parameters of L-BFGS-B
%
opts.print      = 0;
opts.maxIt      = 1000;
opts.tol        = 1e-5;
opts.whichL     = 'wolfeG'; % wolfeG, wolfeB, mt 
opts.storeCOMP  = false;
opts.store      = true;
opts.whichV     = 's'; %'g'
opts.nrmtol     = 'inf'; % '2'
opts.c2         = 0.9;
opts.c1         = 1e-4;
opts.l          = 5;

%
% Decomposition parameter
%
R           = 2;                                % "Rank" of tensor


fprintf('Tensor decomposition: lbfgsb and compact  \n');
fprintf('\n');
fprintf('Line-search: %s          \n',opts.whichL);
fprintf('V-strategy:  %s          \n',opts.whichV);
fprintf('         n=  %i          \n',sz(1)*6);
fprintf('\n');
fprintf('  num. sim:  %i          \n',numsim);
fprintf('         R:  %i          \n',R);
fprintf('------------------------------------------\n');
fprintf('s    \t tme (compac) \t fk (compac)  \t tme (lbfgsb) \t fk (lbfgsb) \t info \n');


%
% Use the TT cp_optim function and most default parameters
% for L-BFGS-B
%

% Data storage
infos = cell(numsim,2);
diff  = zeros(numsim,1); 

infop = '   ';

for s=1:numsim

    prob = create_problem('Size',sz);
    
    %
    % Call the optimization "wrapper" using the compact representation
    %
    method      = 'compLS1';                        % Choice of method
    [M,MO,info] = cp_opt(prob.Data,R,'method',method,opts,'printitn',0);      
    
    %
    % Optimization algorithm using the L-BFGS-B
    %
    method = 'lbfgsb';
    [M1,MO1,info1] = cp_opt(prob.Data,R,'method',method,'init',MO,'printitn',0);

    % Store info
    infos{s,1} = info;
    infos{s,2} = info1;

    diff(s) = abs(info.f-info1.f) > 1e-1;

    if diff(s) == 1
        infop(1) = 'd';
    end
    if info.opttime < info1.opttime
        infop(2) = 'c';
    else
        infop(2) = 'b';
    end

    fprintf('%i \t %0.5e \t %0.5e \t %0.5e \t %0.5e \t %s\n',...
       s,info.opttime,info.f,info1.opttime,info1.f,infop);

    infop(1:3)  = '   ';

end

save([datapath,'/tensordecomp_',num2str(numsim)],'infos','diff');

