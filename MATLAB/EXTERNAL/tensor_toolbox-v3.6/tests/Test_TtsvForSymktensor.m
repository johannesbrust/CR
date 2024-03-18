% Testing ttsv for Symktensor
classdef Test_TtsvForSymktensor < matlab.unittest.TestCase
    methods (Test)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % --- TTSV ---
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function Ttsv(testCase)
            % declaring a standard gaussian random vector to multiply
            x=mvnrnd(zeros(10,1),eye(10,10),1)';
            % decaring parameters for testing
            lambda=mvnrnd(zeros(10,1),eye(10,10),1)'; % std. gaussian vector
            U=mvnrnd(zeros(10,1),eye(10,10),10); % std. gaussian columns of U
            m1=2;
            m2=5;
            m3=6;

            % testing on symktensor of order m1=2
            A = symktensor(lambda,U,m1);
            T=full(full(A));
            % --- Case I: Multiply in all modes but the first ---
            y_A=ttsv(A,x,-1);
            y_T=ttsv(T,x,-1);
            testCase.verifyEqual(y_A,y_T,'RelTol', 1e-10);
            % --- Case II: Multiply in all modes but the first two ---
            y_A=ttsv(A,x,-2);
            y_T=ttsv(T,x,-2);
            testCase.verifyEqual(y_A,y_T,'RelTol', 1e-10);
            % --- Case II: Multiply in all modes ---
            y_A=ttsv(A,x);
            y_T=ttsv(T,x);
            testCase.verifyEqual(y_A,y_T,'RelTol', 1e-10);

            % testing on symktensor of order m2=5
            A = symktensor(lambda,U,m2);
            T=full(full(A));
            % --- Case I: Multiply in all modes but the first ---
            y_A=ttsv(A,x,-1);
            y_T=ttsv(T,x,-1);
            testCase.verifyEqual(y_A,y_T,'RelTol', 1e-10);
            % --- Case II: Multiply in all modes but the first two ---
            y_A=ttsv(A,x,-2);
            y_T=ttsv(T,x,-2);
            testCase.verifyEqual(y_A,y_T,'RelTol', 1e-10);
            % --- Case III: Multiply in all modes but the first three ---
            y_A=full(full(ttsv(A,x,-3)));
            y_T=ttsv(T,x,-3,1);
            testCase.verifyEqual(double(y_A),double(y_T),'RelTol', 1e-10);

            % testing on symktensor of order m3=6
            A = symktensor(lambda,U,m3);
            T=full(full(A));
            % --- Case I: Multiply in all modes but the first ---
            y_A=ttsv(A,x,-1);
            y_T=ttsv(T,x,-1);
            testCase.verifyEqual(y_A,y_T,'RelTol', 1e-10);
            % --- Case II: Multiply in all modes but the first two ---
            y_A=ttsv(A,x,-2);
            y_T=ttsv(T,x,-2);
            testCase.verifyEqual(y_A,y_T,'RelTol', 1e-10);
            % --- Case III: Multiply in all modes but the first four ---
            y_A=full(full(ttsv(A,x,-4)));
            y_T=ttsv(T,x,-4,1);
            testCase.verifyEqual(double(y_A),double(y_T),'RelTol', 1e-10);

        end
    end
end