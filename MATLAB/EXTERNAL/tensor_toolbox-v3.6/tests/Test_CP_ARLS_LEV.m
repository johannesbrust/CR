% Testing cp_arls_lev
classdef Test_CP_ARLS_LEV < matlab.unittest.TestCase
    methods (Test)

        function TensorFibersFunc(testCase)
            rng('default');
            X = tenrand([5 4 3 2]);
            D = ndims(X);
            S = 10;
            for K = 1:4
                XK = double(tenmat(X,K));
                FK = randi(size(XK,2),S,1); % Choose S random columns of XK
                XFKa = XK(:,FK);
                SK = tt_ind2sub64(size(X,[1:K-1,K+1:D]),uint64(FK));
                XFKb = fibers(X,K,SK);
                testCase.verifyEqual(XFKa,XFKb);
            end        
        end

        function SptensorFibersFunc(testCase)
            rng('default');
            X = sptenrand(10*[5 4 3 2], 50);
            D = ndims(X);
            S = 10;
            for K = 1:4
                XK = double(sptenmat(X,K));
                FK = randi(size(XK,2),S,1); % Choose S random columns of XK
                XFKa = XK(:,FK);
                SK = tt_ind2sub64(size(X,[1:K-1,K+1:D]),uint64(FK));
                XFKb = fibers(X,K,SK);
                testCase.verifyEqual(XFKa,XFKb);
            end 
        end

        function BasicSparseTensor(testCase)
            % This test checks whether CP_ARLS_LEV runs with various
            % settings on the input parameters

            % Set up a simple sparse tensor
            R = 2;
            sz = 5*[5 4 3];
            s = 2^17; % Only used for threshold
            info = create_problem('Size', sz, 'Num_Factors', R, ...
                'Factor_Generator','rand','Sparse_Generation', 100);

            % Run with default parameters
            rng("default");
            M = cp_arls_lev(info.Data,R,'printitn',0);

            % Run with different fit methods
            rng("default");
            M = cp_arls_lev(info.Data,R, 'truefit', 'iter','printitn',0);
            rng("default");
            M = cp_arls_lev(info.Data,R, 'truefit', 'final','printitn',0);

            % Run with different least squres and fit samples (should
            % trigger nsampfit warning due to small tensor).
            rng("default");
            M = cp_arls_lev(info.Data,R, 'nsamplsq', 2^16,'printitn',0);
            rng("default");
            M = cp_arls_lev(info.Data,R, 'nsampfit', 10^4,'printitn',0);

            % Run with different termination criteria
            rng("default");
            M = cp_arls_lev(info.Data,R, 'epoch', 6, 'newitol', 4, 'tol', 1e-3,'printitn',0);

            % Run with deterministic inclusion
            rng("default");
            M = cp_arls_lev(info.Data,R, 'thresh', 1.0/s,'printitn',0);
            
            % Run with RRF initialization
            rng("default");
            M = cp_arls_lev(info.Data,R, 'init', 'RRF','printitn',0);

            % Run with different dimorder
            rng("default");
            M = cp_arls_lev(info.Data,R, 'dimorder', [2 1 3],'printitn',0);
        end

        function BasicDenseTensor(testCase)
            R = 2;
            sz = 5*[5 4 3];
            s = 2^17; % Only used for threshold
            info = create_problem('Size', sz, 'Num_Factors', R, 'Noise', 0.0);

            % Run with default parameters
            rng("default");
            M = cp_arls_lev(info.Data,R,'printitn',0);

            % Run with different fit methods
            rng("default");
            M = cp_arls_lev(info.Data,R, 'truefit', 'iter','printitn',0);
            rng("default");
            M = cp_arls_lev(info.Data,R, 'truefit', 'final','printitn',0);

            % Run with different least squres and fit samples 
            rng("default");
            M = cp_arls_lev(info.Data,R, 'nsamplsq', 2^8,'printitn',0);
            rng("default");
            M = cp_arls_lev(info.Data,R, 'nsampfit', 10^2,'printitn',0);

            % Run with different termination criteria
            rng("default");
            M = cp_arls_lev(info.Data,R, 'epoch', 6, 'newitol', 4, 'tol', 1e-3,'printitn',0);

            % Run with deterministic inclusion
            rng("default");
            M = cp_arls_lev(info.Data,R, 'thresh', 1.0/s,'printitn',0);

            % Run with different dimorder
            rng("default");
            M = cp_arls_lev(info.Data,R, 'dimorder', [2 1 3],'printitn',0);
        end

    end % methods
end % classdef Test_CP_ARLS_LEV
