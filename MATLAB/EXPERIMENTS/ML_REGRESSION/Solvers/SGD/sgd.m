function [w,W] = sgd(w,g,W,varargin)
%SGD Stochastic gradient descent
%   [w,W] = sgd(w,g,W,alp) computes the update w = w - alp*g

w = w - varargin{1}*g;

end