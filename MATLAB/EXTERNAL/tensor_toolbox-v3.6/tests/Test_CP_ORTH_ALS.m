% Testing cp_orth_als
classdef Test_CP_ORTH_ALS < matlab.unittest.TestCase
    
    methods (Test)
        
        function Check_Shape_OrthALS(testCase)
            % This test checks for the correctness of the shapes
            % of the tensor returned by CP_ORTH_ALS

            rng('default');

            X = sptenrand([5 4 3], 10);
            P = cp_orth_als(X,2);
            testCase.verifySize(P,[5 4 3]);
           
            X = tendiag([1 1 1]);
            P = cp_orth_als(X,3);
            testCase.verifySize(P,[3 3 3]);

        end

        function Check_Correctness_OrthALS(testCase)
            % This test checks for the correctness of CP_ORTH_ALS

            rng('default');
            
            % two-rank approximation of a 2x2 diagonal matrix with distinct
            % elements
            X = tendiag([1 1]);
            X(1,1) = 2;
            P = cp_orth_als(X,2);
            testCase.verifyEqual(tensor(int8(P.lambda(1)*P.U{1}(:,1)* ...
                P.U{2}(:,1)'+P.lambda(2)*P.U{1}(:,2)*P.U{2}(:,2)')), X);

            % two-rank approximation of a 2x2 diagonal matrix with distinct
            % elements, in hybrid mode
            X = tendiag([1 1]);
            X(1,1) = 2;
            P = cp_orth_als(X,2,'stop_orth',3);
            testCase.verifyEqual(tensor(int8(P.lambda(1)*P.U{1}(:,1)* ...
                P.U{2}(:,1)'+P.lambda(2)*P.U{1}(:,2)*P.U{2}(:,2)')), X);
            
            % two-rank approximation of a 3x3 matrix constructed from
            % fixed vectors u1, u2, v1, v2
            u1 = [1 2 3];
            u2 = [3 2 1];
            v1 = [8 9 0];
            v2 = [5 6 7];
            X = tensor(u1'*v1 + u2'*v2);
            P = cp_orth_als(X,2);
            testCase.verifyEqual(tensor(uint8(P.lambda(1)*P.U{1}(:,1)* ...
                P.U{2}(:,1)'+P.lambda(2)*P.U{1}(:,2)*P.U{2}(:,2)')),X);

            % two-rank approximation of a 3x3 matrix constructed from
            % vectors u1, u2, v1, v2 generated randomly from a discrete
            % uniform distribution of integers
            u1 = randi(5,1,3);
            u2 = randi(10,1,3);
            v1 = randi(15,1,3);
            v2 = randi(20,1,3);
            X = tensor(u1'*v1 + u2'*v2);
            P = cp_orth_als(X,2);
            testCase.verifyEqual(tensor(uint8(P.lambda(1)*P.U{1}(:,1)* ...
                P.U{2}(:,1)'+P.lambda(2)*P.U{1}(:,2)*P.U{2}(:,2)')),X);

            % two-rank approximation of a 3x3 matrix constructed from
            % vectors u1, u2, v1, v2 generated randomly from a discrete
            % uniform distribution of integers, in hybrid mode
            u1 = randi(5,1,3);
            u2 = randi(10,1,3);
            v1 = randi(15,1,3);
            v2 = randi(20,1,3);
            X = tensor(u1'*v1 + u2'*v2);
            P = cp_orth_als(X,2,'stop_orth',4);
            testCase.verifyEqual(tensor(uint8(P.lambda(1)*P.U{1}(:,1)* ...
                P.U{2}(:,1)'+P.lambda(2)*P.U{1}(:,2)*P.U{2}(:,2)')),X);

            % three-rank approximation of a 3x3 matrix constructed from
            % fixed vectors u1, u2, u3, v1, v2, v3
            u1 = [1 2 3];
            u2 = [3 2 1];
            v1 = [8 9 0];
            v2 = [5 6 7];
            u3 = [-1 -5 -6];
            v3 = [3 -2 2];
            X = tensor(u1'*v1 + u2'*v2 + u3'*v3);
            P = cp_orth_als(X,3);
            testCase.verifyEqual(tensor(int8(P.lambda(1)*P.U{1}(:,1)* ...
                P.U{2}(:,1)'+P.lambda(2)*P.U{1}(:,2)*P.U{2}(:,2)'+ ...
                P.lambda(3)*P.U{1}(:,3)*P.U{2}(:,3)')),X);
        
            % three-rank approximation of a 3x3 matrix constructed from
            % vectors u1, u2, u3, v1, v2, v3 generated randomly from a 
            % discrete uniform distribution of integers
            u1 = randi(5,1,3);
            u2 = randi(6,1,3);
            v1 = randi(7,1,3);
            v2 = randi(8,1,3);
            u3 = randi(9,1,3);
            v3 = randi(10,1,3);
            X = tensor(u1'*v1 + u2'*v2 + u3'*v3);
            P = cp_orth_als(X,3);
            testCase.verifyEqual(tensor(int8(P.lambda(1)*P.U{1}(:,1)* ...
                P.U{2}(:,1)'+P.lambda(2)*P.U{1}(:,2)*P.U{2}(:,2)'+ ...
                P.lambda(3)*P.U{1}(:,3)*P.U{2}(:,3)')),X);


            % three-rank approximation of a 3x3 matrix constructed from
            % vectors u1, u2, u3, v1, v2, v3 generated randomly from a 
            % discrete uniform distribution of integers, while running
            % OrthALS in hybrid mode
            u1 = randi(5,1,3);
            u2 = randi(6,1,3);
            v1 = randi(7,1,3);
            v2 = randi(8,1,3);
            u3 = randi(9,1,3);
            v3 = randi(10,1,3);
            X = tensor(u1'*v1 + u2'*v2 + u3'*v3);
            P = cp_orth_als(X,3,'stop_orth',3);
            testCase.verifyEqual(tensor(int8(P.lambda(1)*P.U{1}(:,1)* ...
                P.U{2}(:,1)'+P.lambda(2)*P.U{1}(:,2)*P.U{2}(:,2)'+ ...
                P.lambda(3)*P.U{1}(:,3)*P.U{2}(:,3)')),X);
        end

        function Check_Params(testCase)
            % This test checks whether CP_ORTH_ALS runs with various
            % settings on the input parameters

            rng('default');

            X = sptenrand([5 4 3], 10);
            
            P = cp_orth_als(X,2);

            P = cp_orth_als(X,2,'stop_orth',5);
            P = cp_orth_als(X,2,'stop_orth',15);

            P = cp_orth_als(X,2,'tol',1.0e-2);
            P = cp_orth_als(X,2,'tol',1.0e-2,'stop_orth',3);

            P = cp_orth_als(X,2,'maxiters',2);
            P = cp_orth_als(X,2,'tol',1.0e-2,'maxiters',2);
            
            P = cp_orth_als(X,2,'dimorder',[1 3 2]);
            P = cp_orth_als(X,2,'dimorder',[2 1 3]);
            P = cp_orth_als(X,2,'dimorder',[2 3 1]);
            P = cp_orth_als(X,2,'dimorder',[3 1 2]);
            P = cp_orth_als(X,2,'dimorder',[3 2 1]);

            P = cp_orth_als(X,2,'dimorder',[3 2 1],'init','nvecs');
            U0 = {rand(5,2),rand(4,2),rand(3,2)}; %<-- Initial guess for factors of P
            [~,~,out] = cp_orth_als(X,2,'dimorder',[3 2 1],'init',U0);
            
            P = cp_orth_als(X,2,out.params); %<-- Same params as previous run
            
            P = cp_orth_als(X,2,'printitn',0); %<-- Do not print fit
            P = cp_orth_als(X,2,'printitn',2); %<-- Print fit every 2 iterations
            P = cp_orth_als(X,2,'printitn',2,'stop_orth',8); %<-- Print fit every 2 iterations
    
        end
    end % methods
end % classdef Test_CP_ORTH_ALS
