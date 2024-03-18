function [X, stat] = cp_spm(T, varargin)
%CP_SPM Symmetric tensor decomposition using Subspace Power Method (SPM).
%
%   MODEL = CP_SPM(X) fits an symmetric CP model to the symmetric tensor X 
%   using the SPM algorithm, described in the reference below. The result 
%   MODEL is a symmetric Kruskal tensor (symktensor). Its rank is estimated 
%   by the algorithm.
%   
%   MODEL = CP_SPM(X,R) fits an R-component symmetric CP model to the
%   tensor X using SPM. Set R = [] to have the rank estimated by the
%   algorithm
%
%   [M, INFO] = CP_SPM(X, ...) returns additional information in INFO.
%
%   [...] = CP_SPM(X,[R],'param','value') takes additional arguments:
%
%      'rank_sel' - If R is not provided, provide this option to select the 
%              rank based on the eigenvalues of the flattening. Accepted 
%              values are functions (that select the rank given the
%              eigenvalues) or scalars, in that case this is a threshold;
%              the rank is all the values above this threshold. Default is
%              1e-4.
%      'gradtol' - Gradient tolerance. The power method finishes if the
%              norm of the gradient is smaller than this value.
%      'maxtries' - Maximum number of power method iterations.
%      'restart_tol' - If the function value after the power method is
%              less than 1 - restart_tol, restart the power method.
%      'ntries' - Maximum number of power method restarts.
%      'reinit_tol' - If the function value is less than this, reinitialize 
%              the power_method. This option is added because when the rank
%              of the tensor is less than the dimension, we might very
%              rarely pick a starting point that is very close to
%              orthogonal to the remaining a_i. In this case, the power
%              method is stuck in an almost saddle point, and it is faster
%              to just reinitialize the power method.
%                    
%
%   Reference:
%   * J. Kileel, J. M. Pereira, Subspace power method for symmetric tensor
%                           decomposition and generalized PCA
%   
%   See also the SPM repository:
%     <a href="https://github.com/joaompereira/SPM">https://github.com/joaompereira/SPM</a>
%
%
%   <a href="matlab:web(strcat('file://',fullfile(getfield(what('tensor_toolbox'),'path'),'doc','html','cp_spm_doc.html')))">Additional documentation for CP-SPM</a> 
% 
%   See also CP_SYM.
%% 
%Tensor Toolbox for MATLAB: <a href="https://www.tensortoolbox.org">www.tensortoolbox.org</a>
    
    % Ensure Tensor is symmetric, symmetrizing if needed
    if ismember(class(T), {'tensor','tenmat','ttensor','sptensor',...
                           'sumtensor','ktensor', 'symktensor'})
        T = full(T);
        if class(T)~="symktensor"
            T = symtensor(T);
        end
    end
        
    n = size(T,1);
    nd = ndims(T);

    nd2 = ceil(nd/2);

    assert(nd > 2);

    %% Parse optional parameters
    params = inputParser;
    params.addOptional('rank', [], @(x) isempty(x) || x>0);
    params.addParameter('maxiter', 5000, @(x) x>0);
    params.addParameter('ntries' ,    3, @(x) x>0);
    params.addParameter('gradtol',1e-14, @(x) x>0);
    params.addParameter('rank_sel', 1e-4);
    params.addParameter('reinit_tol', 5e-3/sqrt(n), @(x) x>0);
    params.addParameter('restart_tol', 1e-3, @(x) x>0);
    params.parse(varargin{:});

    opts = params.Results;
    r = opts.rank;
                                
    timer = tic;

    %% Extract Subspace 
    % This part of the algorithm differs slightly if nd is even or odd
    if mod(nd,2)
        % We calculate the flattening of the tensor, but take out repeated
        % rows and columns. 
        % These next variables are useful to map unique rows to all rows
        [symind_l, findsym_l, symindscale_l] = symmetric_indices(n, nd2);
        [symind_r, findsym_r, symindscale_r] = symmetric_indices(n, nd2-1);
        
        symindscale_l = sqrt(symindscale_l);
        symindscale_r = sqrt(symindscale_r);
        findsym_l = reshape(findsym_l,[],1);
        findsym_r = reshape(findsym_r,[],1);

        dl = size(symind_l, 1);
        dr = size(symind_r, 1);
        
        % This is a piece of code that maps unique indices of the nd
        % tensor to the flattening without repeated rows/columns
        flat_indices = reshape(cat(2, symind_l, zeros(dl, nd2-1)), dl, 1, nd) + ...
                reshape(cat(2, zeros(dr, nd2), symind_r), 1, dr, nd);
        fi_sorted = sort(flat_indices, 3, 'ascend');
        fi_sim = nchoosek(n+nd-1, nd) + zeros(dl, dr);
        v = (0:n-1); 
        for i=1:nd
            if i>1
                v = cumsum(v); 
            end
            fi_sim = fi_sim - v(n+1 - fi_sorted(:, :, nd + 1 - i));
        end

        flat_T = T.val(fi_sim);
        % To keep the same subspace, we rescale rows and
        % columns appropriately
        flat_T = symindscale_l.* flat_T .* symindscale_r';
        
        % Singular value decomposition
        [symV, D, symU] = svd(flat_T, 'econ');
        D = diag(D);
        
        % Determine tensor rank by the singular values of mat(T)
        if isempty(r)
            r = rank_selector(D, opts.rank_sel);
            if isempty(r)
                X = [];
                stat = struct();
                return
            end
        end
        
        C = diag(1./D(1:r));

        V = symV(:,1:r)./symindscale_l;
        V = V(findsym_l, :);
        
        U = symU(:,1:r)./symindscale_r;
        U = U(findsym_r, :);
        
        
    else
        % We calculate the flattening of the tensor, but take out repeated
        % rows and columns. 
        % These next variables are useful to map unique rows to all rows
        [symind, findsym, symindscale] = symmetric_indices(n, nd2);

        symindscale = sqrt(symindscale);
        findsym = reshape(findsym,[],1);
        
        dsym = size(symind, 1);
    
        % This is a piece of code that maps unique indices of the nd
        % tensor to the flattening without repeated rows/columns
        flat_indices = reshape(cat(2, symind, zeros(dsym, nd2)), dsym, 1, nd) + ...
                reshape(cat(2, zeros(dsym, nd2), symind), 1, dsym, nd);
        fi_sorted = sort(flat_indices, 3, 'ascend');
        fi_sim = nchoosek(n+nd-1, nd) + zeros(dsym, dsym);
        v = (0:n-1); 
        for i=1:nd
            if i>1
                v = cumsum(v); 
            end
            fi_sim = fi_sim - v(n+1 - fi_sorted(:, :, nd + 1 - i));
        end

        flat_T = T.val(fi_sim);
        % To keep the same subspace, we rescale rows and
        % columns appropriately
        flat_T = (symindscale.*symindscale') .* flat_T;

        % Eigen decomposition
        [symV, D] = eig(flat_T, 'vector');
        [~, I] = sort(abs(D), 'descend');
        D = D(I);

        % Determine tensor rank by the eigenvalues of mat(T)
        if isempty(r)
            r = rank_selector(D, opts.rank_sel);
        end

        C = diag(1./D(1:r));

        V = symV(:,I(1:r))./symindscale;
        V = V(findsym, :);
    end

    % Pre-allocation of X and lambda
    A = zeros(n,r);
    if nargout>1
        lambda = zeros(1,r);
    end

    % C_n from Lemma 4.7
    if nd2<=4
      cn = sqrt(2*(nd2-1)/nd2);
    else
      cn = (2-sqrt(2))*sqrt(nd2);
    end

    lap = toc(timer);
    stat.extracttime = lap;
    stat.powertime = 0;
    stat.deflatetime = 0;
    stat.avgiter = 0;
    stat.nrr = 0;

    for k = r:-1:1

        %% Power Method        
            
        V_ = reshape(V,[],n*k);

        for tries = 1:opts.ntries
          
          % Initialize Xk
          Ak = randn(n,1);
          Ak = Ak/norm(Ak);
        
          for iter = 1:opts.maxiter

            % Calculate power of Xk
            Apow = Ak;
            for i=2:nd2-1
                Apow = reshape(Apow*Ak',[],1);
            end

            % Calculate contraction of V with x^(n2-1)
            VAk = reshape(Apow'*V_,n,k);
                        
            Ak_new = VAk*(Ak'*VAk)';

            f = Ak_new'*Ak;

            % Determine optimal shift
            % Sometimes due to numerical error f can be greater than 1
            f_ = max(min(f,1),.5);
            clambda = sqrt(f_*(1-f_));
            shift = cn*clambda;

            if f < opts.reinit_tol
                % Xk was not a good initialization
                % Initialize it again at random
                Ak = randn(n,1);
                Ak = Ak/norm(Ak);
            else
                % Shifted power method
                Ak_new = Ak_new + shift*Ak;
                Ak_new = Ak_new/norm(Ak_new);

                if norm(Ak - Ak_new) < opts.gradtol
                    % Algorithm converged
                    Ak = Ak_new;
                    break
                else
                    Ak = Ak_new;
                end
            end
          end 
          
          stat.avgiter = stat.avgiter + iter;
          
          if 1-f<opts.restart_tol
             break
          elseif tries==1 || f>f_
              stat.nrr = stat.nrr + 1;
              f_ = f;
              Ak_ = Ak;
          else
              stat.nrr = stat.nrr + 1;
              Ak = Ak_;
          end
          
        end
        
        timenow = toc(timer);
        stat.powertime = stat.powertime + timenow - lap;
        lap = timenow;
        
        %% Deflation
        % This part of the algorithm differs slightly if nd is even or odd 
        if mod(nd,2)

            % Calculate power of Xk
            Apow = Ak;
            for i=2:nd2-1
                Apow = reshape(Apow*Ak',[],1);
            end

            % Calculate projection of Xpow in subspace
            alphaU = (Apow'*U)';
            
            Apow = reshape(Apow*Ak',[],1);
            
            alphaV = (Apow'*V);

            % Solve for lambda
            D1alphaU = C*alphaU;
            D1alphaV = (alphaV * C)';
            lambdak = 1/(alphaV*D1alphaU);

            if k > 1
                % Calculate the new matrix D and the new subspace

                % Use Householder reflection to update U and C
                x = get_hh_reflector(D1alphaU);

                C = LHR(C,x);
                V = RHR(V,x);
                
                % Use Householder reflection to update V and C
                x = get_hh_reflector(D1alphaV);
                
                C = RHR(C,x);
                U = RHR(U,x);

            end
            
        else
            
            % Calculate power of Xk
            Apow = Ak;
            for i=2:nd2
                Apow = reshape(Apow*Ak',[],1);
            end

            % Calculate projection of Xpow in subspace
            alpha = (Apow'*V)';

            % Solve for lambda
            D1alpha = C*alpha;
            lambdak = 1/(alpha'*D1alpha);

            if k > 1
                % Calculate the new matrix D and the new subspace

                % Use Householder reflection to update V and C
                x = get_hh_reflector(D1alpha);

                C = RHR(LHR(C,x),x);

                V = RHR(V,x);

            end
            
            
        end
        
        A(:,k) = Ak;
        lambda(k) = lambdak;
                
        timenow = toc(timer);
        stat.deflatetime = stat.deflatetime + timenow - lap;
        lap = timenow;

    end

    X = symktensor([lambda(:); A(:)],nd, r);
    
    stat.avgiter = stat.avgiter/r;
    stat.totaltime = toc(timer);        
        
end

function [symind, findsym, ncomb] = symmetric_indices(d, n)
% Map indices of symmetric tensor to indices of full tensor

    % Number of rows and columns of new matrix
    dsym = nchoosek(d+n-1,n);

    % All the indices of symmetric tensor
    symind = nchoosek(1:d+n-1,n)-(0:n-1);
    symind = symind(:,end:-1:1);

    % Positions of all repeated indices
    if n==1
        findsym= zeros(d,1);
    else
        findsym = zeros(d * ones(1,n));
    end
    findsym((symind-1) * (d.^(0:n-1)') + 1) = 1:dsym;
    perm = 1:n;
    for i=2:n
        findsym_i = findsym;
        for k=2:i
            perm([n-i+1,n-i+k]) = [n-i+k,n-i+1];
            findsym = max(findsym, permute(findsym_i, perm));
            perm([n-i+1,n-i+k]) = [n-i+1,n-i+k];
        end
    end

    % ncomb is number of repeated indices
    S = ones(dsym,1);
    ncomb = factorial(n) * S;

    for i=2:n
        S(symind(:,i-1)~=symind(:,i)) = 0;
        S = S + 1;
        ncomb = ncomb ./ S;
    end
    
end

function r = rank_selector(D, rank_sel)
% Select rank by looking at eigenvalues

    if isa(rank_sel,'function_handle')
        r = rank_sel(D);
    elseif isscalar(rank_sel)
        typical = mean(D.^2) / mean(abs(D));
        r = sum(abs(D) > rank_sel * typical);
    elseif rank_sel == "plot_only"
        plot(abs(D))
        r = [];
    else
        error('Rank selector option not implemented yet')
    end

end

function y = get_hh_reflector(y)
% Get vector for Householder reflection
%    The last column of the corresponding Householder reflection, is a 
%    multiple of the input y.

    norm_y = norm(y);
    y(end) = y(end) + norm_y*sign(y(end));
    y = y / sqrt(abs(y(end))*norm_y);

end

function A = LHR(A,x)
% Apply Householder reflection from the left

A = A(1:end-1, :) + x(1:end-1) * (-x'*A);

end

function A =  RHR(A,x)
% Apply Householder reflection from the right

A = A - (A*x)*x';
A = A(:, 1:end-1);

end

