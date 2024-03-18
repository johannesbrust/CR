function [step,x1,f1,g1,jf,iExit] = armijoG1(func,step,stepMax,f,g,p,x)

%        [step,x1,f1,g1,jf,iExit] = armijoG1(func,step,f,g,p,x)
%
%  armijoG1  finds a step along the search direction  p,  such that
%  the function  f  is sufficiently reduced, i.e.,
%            f(x + step*p)  <  f(x).
%
%  On entry,
%  step     is the initial estimate of the step.
%  f,g      are the function and gradient.
%  x        is the base point for the search.
%  p        is the search direction.
%
%  Output parameters...
%
%  step     is the final step.
%  x1       is the new point.
%  f1       is the function value at x1.
%  g1       is the gradient       at x1.
%
%  iExit    Result
%  -----    ------
%    0      Repeat the search with smaller stepMax.
%    1      The search is successful.
%    2      The search is successful with the initial step.
%    3      A better point was found but too many functions
%           were needed (not sufficient decrease).
%    7      Too many function calls.
%    8      No descent direction
%==========================================================================
% 02/11/24, J.B., the "G" represents an implementation of P.E. Gill
% 02/21/24, J.B., update of the function interface

armijoTol = 1e-4 ; % 1e-4
gammaC    = 0.5       ;  % contraction factor
jfmax     = 20        ;  % max functions per line search

step0     = step;
f1        = f;   g1 = g;   x1 = x;

gp        = g'*p;

if gp >= 0
  jf     = 0;
  inform = 8; % No descent direction
else
  jf    =  1;
  while jf <= jfmax
    x1 = x + step*p;

    % f1      = func(x1);
    % g1      = grad(x1);

    [f1,g1]   = func(x1);

    if f1 <= f + armijoTol*step*gp, break; end

    jf   =  jf + 1;
    step  =  gammaC*step;  %     Reduce the step.
  end

  if jf <= jfmax
    if step == stepMax
      inform = 2;
    else
      inform = 1;
    end
  elseif f1 < f
    inform = 3;
  else
    inform = 7;
  end
end

iExit = inform;

