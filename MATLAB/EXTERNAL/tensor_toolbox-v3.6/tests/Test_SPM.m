classdef Test_SPM < matlab.unittest.TestCase
    
    properties (TestParameter)
        nd = struct( 'three', 3, 'four', 4);
        dim = struct( 'six', 6, 'ten', 10, 'twenty', 20);
        rank = struct( 'one', 1, 'five', 5, 'ten', 10)
        gen = struct( 'rand', @rand, 'randn', @randn);
        lambda_gen = struct( 'ones', @ones, ...
                             'ber', @(varargin) 2*randi(2, varargin{:}) - 3);
    end
    
    methods (Test)
        function Accurate_Decomposition(testCase, nd, dim, rank, gen, lambda_gen)
            if rank > nchoosek(dim + floor(nd/2) - 1, floor(nd/2))
                return
            elseif func2str(gen)=="rand" && func2str(lambda_gen) ~= "ones"
                return
            end

            A = gen(dim, rank);
            lambda = lambda_gen(rank, 1);
            T = full(symktensor([lambda(:); A(:)], nd, rank));
            A_est = cp_spm(T);
            T_est = full(A_est);

            testCase.verifyLessThanOrEqual(norm(T(:)-T_est(:)), 1e-6);
            

        end
    end
    
    
    
end