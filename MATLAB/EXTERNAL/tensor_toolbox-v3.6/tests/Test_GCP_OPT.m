% Testing gcp_opt
classdef Test_GCP_OPT < matlab.unittest.TestCase
    methods (Test)

        function SparseBinary(testCase)
            [output,X] = evalc('create_problem_binary([10 15 20],3);');
            [output,M] = evalc('gcp_opt(X,10,''type'',''binary'',''maxiters'',2);');
            testCase.verifyTrue(contains(output, 'End Main Loop'));
        end % SparseBinary
        
        function FunctionTypeNormal(testCase)
            % normal, gaussian
            X = [-1,1]'; M = [-1.1,1.1]';
            [fh,gh,lb]=tt_gcp_fg_setup('normal',X);
            testCase.verifyEqual(fh(X,M),[ 0.01 0.01]', 'RelTol', 1e-8);
            testCase.verifyEqual(gh(X,M),[-0.20 0.20]', 'RelTol', 1e-8);
            testCase.verifyEqual(lb,-Inf);
            [fh,gh,lb]=tt_gcp_fg_setup('gaussian',X);
            testCase.verifyEqual(fh(X,M),[ 0.01 0.01]', 'RelTol', 1e-8);
            testCase.verifyEqual(gh(X,M),[-0.20 0.20]', 'RelTol', 1e-8);
            testCase.verifyEqual(lb,-Inf);
        end % FunctionTypeNormal
        
        function FunctionTypeBinary(testCase)
            % binary, bernoulli-odds
            X = [0,1]'; M = [0.1,0.9]';
            [fh,gh,lb]=tt_gcp_fg_setup('binary',X);
            testCase.verifyEqual(fh(X,M),[0.095310179804325  0.747214401719110]', 'RelTol', 1e-8);
            testCase.verifyEqual(gh(X,M),[0.909090909090909 -0.584795321513970]', 'RelTol', 1e-8);
            testCase.verifyEqual(lb,0);
            [fh,gh,lb]=tt_gcp_fg_setup('bernoulli-odds',X);
            testCase.verifyEqual(fh(X,M),[0.095310179804325  0.747214401719110]', 'RelTol', 1e-8);
            testCase.verifyEqual(gh(X,M),[0.909090909090909 -0.584795321513970]', 'RelTol', 1e-8);
            testCase.verifyEqual(lb,0);
        end % FunctionTypeNBinary
        
        function FunctionTypeBernoulliLogit(testCase)
            % bernoulli-logit
            X = [0,1]'; M = [0.1,0.9]';
            [fh,gh,lb]=tt_gcp_fg_setup('bernoulli-logit',X);
            testCase.verifyEqual(fh(X,M),[0.744396660073571  0.341153874732088]', 'RelTol', 1e-8);
            testCase.verifyEqual(gh(X,M),[0.524979187478940 -0.289050497374996]', 'RelTol', 1e-8);
            testCase.verifyEqual(lb,-Inf);
        end % FunctionTypeBernoulliLogit
        
        function FunctionTypeCount(testCase)
            % count, poisson
            X = [1,2]'; M = [1.1,2.1]';
            [fh,gh,lb]=tt_gcp_fg_setup('count',X);
            testCase.verifyEqual(fh(X,M),[1.004689820104766 0.616125310446007]', 'RelTol', 1e-8);
            testCase.verifyEqual(gh(X,M),[0.090909090991736 0.047619047664399]', 'RelTol', 1e-8);
            testCase.verifyEqual(lb,0);
            [fh,gh,lb]=tt_gcp_fg_setup('poisson',X);
            testCase.verifyEqual(fh(X,M),[1.004689820104766 0.616125310446007]', 'RelTol', 1e-8);
            testCase.verifyEqual(gh(X,M),[0.090909090991736 0.047619047664399]', 'RelTol', 1e-8);
            testCase.verifyEqual(lb,0);
        end % FunctionTypeCount
        
        function FunctionTypePoissonLog(testCase)
            % poisson-log
            X = [1,2]'; M = [1.1,2.1]';
            [fh,gh,lb]=tt_gcp_fg_setup('poisson-log',X);
            testCase.verifyEqual(fh(X,M),[1.904166023946433 3.966169912567650]', 'RelTol', 1e-8);
            testCase.verifyEqual(gh(X,M),[2.004166023946433 6.166169912567650]', 'RelTol', 1e-8);
            testCase.verifyEqual(lb,-Inf);
        end % FunctionTypePoissonLog
        
        function FunctionTypeRayleigh(testCase)
            % rayleigh
            X = [1.5,2.5]'; M = [1.6,2.6]';
            [fh,gh,lb]=tt_gcp_fg_setup('rayleigh',X);
            testCase.verifyEqual(fh(X,M),[1.630298613078723 2.637167641737781]', 'RelTol', 1e-8);
            testCase.verifyEqual(gh(X,M),[0.387135806897989 0.210657883371910]', 'RelTol', 1e-8);
            testCase.verifyEqual(lb,0);
        end % FunctionTypeRayleigh
        
        function FunctionTypeGamma(testCase)
            % gamma
            X = [1.5,2.5]'; M = [1.6,2.6]';
            [fh,gh,lb]=tt_gcp_fg_setup('gamma',X);
            testCase.verifyEqual(fh(X,M),[1.407503629249642 1.917049906567377]', 'RelTol', 1e-8);
            testCase.verifyEqual(gh(X,M),[0.039062500034180 0.014792899421939]', 'RelTol', 1e-8);
            testCase.verifyEqual(lb,0);
        end % FunctionTypeGamma
        
        function FunctionTypeHuber(testCase)
            % huber(0.315)
            X = [-1,1]'; M = [-3,1.1]';
            [fh,gh,lb]=tt_gcp_fg_setup('huber (0.315)',X);
            testCase.verifyEqual(fh(X,M),[1.1607750 0.01]', 'RelTol', 1e-8);
            testCase.verifyEqual(gh(X,M),[-0.630000 0.20]', 'RelTol', 1e-8);
            testCase.verifyEqual(lb,-Inf);
        end % FunctionTypeHuber
        
        function FunctionTypeNegativeBinomial(testCase)
            % negative-binomial (2)
            X = [1,2]'; M = [1.1,2.1]';
            [fh,gh,lb]=tt_gcp_fg_setup('negative-binomial (2)',X);
            testCase.verifyEqual(fh(X,M),[2.130501854292898 3.041733756410410]', 'RelTol', 1e-8);
            testCase.verifyEqual(gh(X,M),[0.519480519563164 0.015360983148270]', 'RelTol', 1e-8);
            testCase.verifyEqual(lb,0);
        end % FunctionTypeNegativeBinomial
        
        function FunctionTypeBeta(testCase)
            % beta (1.5)
            X = [1,1.2]'; M = [1.5,1.5]';
            [fh,gh,lb]=tt_gcp_fg_setup('beta (1.5)',X);
            testCase.verifyEqual(fh(X,M),[-1.224744258978328 -1.714642207551294]', 'RelTol', 1e-8);
            testCase.verifyEqual(gh(X,M),[0.4082482905319040  0.244948974351803]', 'RelTol', 1e-8);
            testCase.verifyEqual(lb,0);
        end % FunctionTypeNormal
        
    end % methods
end % classdef Test_GCP_OPT
