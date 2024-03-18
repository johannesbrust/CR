% Testing of squashing out empty slices in a sparse tensor
classdef Test_SptensorSquash < matlab.unittest.TestCase
    methods (Test)

        function Squash(testCase)
            subs = [1 1 1; 3 3 3; 5 5 5];
            vals = [1; 3; 5];
            sz = [5 5 5];
            X = sptensor(subs, vals, sz);
            Y = squash(X);
            testCase.verifyEqual(Y(1,1,1), 1);
            testCase.verifyEqual(Y(2,2,2), 3);
            testCase.verifyEqual(Y(3,3,3), 5);
        end

        function SquashWithMap(testCase)
            subs = [1 1 1; 3 3 3; 5 5 5];
            vals = [1; 3; 5];
            sz = [5 5 5];
            X = sptensor(subs, vals, sz);
            [Y,S] = squash(X);
            testCase.verifyEqual(Y(1,1,1), 1);
            testCase.verifyEqual(Y(2,2,2), 3);
            testCase.verifyEqual(Y(3,3,3), 5);
            for i=1:3
              testCase.verifyEqual(sum(X.subs(:,i) == S{i}(Y.subs(:,i))), 3);
            end
        end

    end
end
