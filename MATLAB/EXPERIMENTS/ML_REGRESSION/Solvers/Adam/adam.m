function [w,W] = adam(w,g,W,varargin)
%ADAM Adaptive moment estimation
%   [w,W] = sgd(w,g,W,varargin) computes the update 
%
%   m   = bet1*m + (1-bet1)*g
%   v   = bet2*v + (1-bet2)*(g.*g)
%   mh  = m ./ (1-bet1^k)
%   vh  = v ./ (1-bet2^k)
%   w   = w - alp* (mh ./ sqrt(vh + eps))

%
% Parameter initializations
%
alp     = varargin{1};
bet1    = varargin{2};
bet2    = varargin{3};
k       = varargin{4};
eps1    = varargin{5};

%
% Update recurrences
%

m       = bet1*W(:,1) + (1-bet1)*g;
v       = bet2*W(:,2) + (1-bet2)*(g.*g);
mh      = m ./ (1-bet1^k);
vh      = v ./ (1-bet2^k);

W(:,1)  = m;
W(:,2)  = v;
w       = w - alp* (mh ./ sqrt(vh + eps1));


end