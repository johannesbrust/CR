%% Orthogonalized Alternating least squares for CANDECOMP/PARAFAC (CP) Decomposition
%
% <html>
% <p class="navigate">
% &#62;&#62; <a href="index.html">Tensor Toolbox</a> 
% &#62;&#62; <a href="cp.html">CP Decompositions</a> 
% &#62;&#62; <a href="cp_orth_als_doc.html">Orth-ALS</a>
% </p>
% </html>
%
% The function |cp_orth_als| is a modification of the standard CP-ALS
% method by orthogonalizing the factor matrices before each iteration. The
% parameter |'stop_orth'| can be used to stop the orthogonalization after a
% fixed number of iterations. 
%
% *IMPORTANT NOTE:* This method may not converge and may not produce
% orthgonal factors. It is a specialized method for a specific purpose.  
%
% Orthogonalized ALS (Orth-ALS) is a modification of the ALS approach that
% provably recovers the true factors with random initialization under
% standard incoherence assumptions on the factors of the tensor. This
% algorithm is a modification of the ALS algorithm to "orthogonalize" the
% estimates of the factors before each iteration. 
% Intuitively, this periodic orthogonalization prevents multiple recovered 
% factors from "chasing after" the same true factors, potentially allowing for the avoidance
% of local optima and more rapid convergence to the true factors.
% 
% Refer to the paper for more details:
% 
% * V. Sharan and G. Valiant. 
% Orthogonalized ALS: A theoretically principled tensor decomposition algorithm 
% for practical use. In International Conference on Machine Learning, 2017.

%% Recommendation
% In practice, we have observed that it is sometimes useful to orthogonalize
% the factor estimates for a few iterations, and then continue with standard ALS. 
% This is true when factors have a high correlation with each other, such as in 
% low-dimensional settings. Our advice to practitioners would be to tune the number of steps 
% for which orthogonalization takes place to get the best results.

%% Example performance on noise-free problem with nearly orthogonal matrices
% Standard normal factor matrices with $r \ll n$ are incoherent and meet
% the conditions of the theory. We see that |cp_orth_als| gets a better fit
% than |cp_als| and is faster than |cp_opt|.

% Setting up the problem
rng('default') 
n = 100; r = 10;
U = cell(3,1);
for k = 1:3
    U{k} = (1/sqrt(n))*randn(n,r);
end
Mtrue = ktensor(U);
Xeasy = full(Mtrue);

%%
% Running |cp_orth_als| yields the desired solution. 

rng('default') 
tic
M = cp_orth_als(Xeasy,r);
toc

%%
% Running |cp_als| does not yield the desired solution.

rng('default') 
tic
M = cp_als(Xeasy,r);
toc

%%
% Running |cp_orth_als| for just 2 iterations "fixes" |cp_als| and even
% yields an improved fit compared to the default of orthogonalizing at
% every iteration. 
rng('default') 
tic
M = cp_orth_als(Xeasy,r,'stop_orth',2);
toc

%% Example performance on Amino Acids test problem.
% We use the well-known _amino acids data set_ from Andersson and Bro.
% It contains fluorescence measurements of 5 samples containing 3 amino
% acids: Tryptophan, Tyrosine, and Phenylalanine.Each amino acid
% corresponds to a rank-one component. The tensor is of size 5 x 51 x 201
% from  5 samples, 51 excitations, and 201 emissions. 
% Further details can be found here: 
% <http://www.models.life.ku.dk/Amino_Acid_fluo>.
% Please cite the following paper for this data: 
% Rasmus Bro, PARAFAC: Tutorial and applications, Chemometrics and 
% Intelligent Laboratory Systems, 1997, 38, 149-171.  
% This dataset can be found in the |doc| directory.
load aminoacids

%%
% We know that |cp_als| can solve this problem, and the final "fit" of 0.97
% is evidence of this.
rng('default') 
tic
M = cp_als(X,3); 
toc

%% 
% The |cp_orth_als| method does not fare as well. The standard CP-ALS
% method guarantees that the fit (f) is non-decreasing. But the CP-ORTH-ALS
% does not have that property, and we can see that the f-value actually
% decreases in some steps due to the orthogonalization.
rng('default') 
tic
M = cp_orth_als(X,3);
toc

%%
% Running |cp_orth_als| with just 10 iterations of orthogonaliztion speeds
% starts off with the same problem but recovers once it resumes regular ALS
% iterations.
rng('default') 
tic
M = cp_orth_als(X,3,'stop_orth',10);
toc




