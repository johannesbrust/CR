%---------------------- mnist_classifier ---------------------------------%
%
% Implementation of an mnist logistic classifier
%
% The objective is 
%
% min_w sum( ln( sum_k exp(wk'x) - wdx'x  ) ),
%
% where w = [w0, ... w9], wk is a (28^2 x 1) vector and x is also
% (28^2 x 1).
%
%-------------------------------------------------------------------------%
% 01/19/23, J.B., initial implementation