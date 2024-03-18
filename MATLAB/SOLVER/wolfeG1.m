function [step,x1,f1,g1,numf,iExit] ...
              = wolfeG1(func,wolfeTols,step,stepMax,f,g,p,x)

%        [step,x1,f1,g1,numf,iExit] =
%               wolfeG1(prob,wolfeTols,step,stepMax,f,g,p,x)
%
%  wolfeG1  finds a step along the search direction  p,  such that
%  the function  f  is sufficiently reduced, i.e.,
%            f(x + step*p)  <  f(x).
%
%  On entry,
%  func          objective function, func(x).
%  grad          gradient function, grad(x).
%  step          is the initial estimate of the step.
%  stepMax       is the maximum allowable step.
%  wolfeTols(1)  Line search accuracy parameter in the range (0,1).
%                0.001 means very accurate.   0.99 means quite sloppy.
%  f,g           are the function and gradient at x.
%  x             is the base point for the search.
%  p             is the search direction.
%
%  Output parameters...
%
%  step          is the final step.
%  x1            is the new point.
%  f1            is the function value at x1.
%  g1            is the gradient at x1.
%
%  iExit    Result
%  -----    ------
%    0      Repeat the search with smaller stepMax.
%    1      The search is successful and step < stepMax.
%    2      The search is successful and step = stepMax.
%    3      A better point was found but too many functions
%           were needed (no sufficient decrease obtained).
%    4      stepMax < tolabs (too small to do a search).
%    5      is not used
%
%    6      No useful step.
%           The interval of uncertainty is less than 2*tolabs.
%           The minimizer is very close to step = zero
%           or the gradients are not sufficiently accurate.
%    7      Too many function calls.
%    8      Bad input parameters
%           (stepMax le toltny  or  oldg ge 0).
%==================================================================
% 02/11/24, J.B., the "G" represents an implementation of P.E. Gill
% 02/21/24, J.B., Modification of the function interface

wolfeGtol = wolfeTols(1);
wolfeFtol = wolfeTols(2);

eps0      = eps^(4/5);

stpmax    = stepMax;
maxf      = 50; % 10

% Define the line search tolerances.

[tolabs,tolrel,toltny,epsaf] = tols( stepMax,f,p,x );

gp       = g'*p;

oldf     = f;            oldg = gp;
f1       = f;            gp1  = gp;
g1       = g;
x1       = x;

first    = 1;
sbest    = 0;
f0       = 0;                               % the shifted objective

g0       =  (1          - wolfeFtol)*oldg;  % the shifted gradient
targetgL =  (wolfeGtol  - wolfeFtol)*oldg;
targetgU = -(wolfeGtol  + wolfeFtol)*oldg;

numf     =  0;
ftry     = f0;
gtry     = g0;

%------------------------------------------------------------------
% Commence main loop, entering srchfg two or more times.
% first = 1 for the first entry,  0 for subsequent entries.
% done  = 1 indicates termination, in which case inform gives
% the result of the search (with inform = iExit as above).
%------------------------------------------------------------------
while  (1)

  [first,improved,done,inform,numf,step,sbest] = ...
      srchfg ( first,                            ...
               maxf, numf,                       ...
               stpmax, epsaf,                    ...
               f0, g0, targetgL, targetgU,       ...
               ftry, gtry,                       ...
               tolabs, tolrel, toltny,           ...
               step );
  if  improved
    f1 = f2;    g1 = g2;
  end

  %---------------------------------------------------------------
  %    done = false  first time through.
  % If done = false, the functions must be computed for the next
  %    entry to srchfg.
  % If done = true,  this is the last time through and inform ge 1.
  %---------------------------------------------------------------
  if  done
    break;
  else
    x2      = x + step*p;
    % f2      = func(x2);
    % g2      = grad(x2);
    %[f2,g2] = prob.obj(x2);
    [f2,g2] = func(x2);
    gp2     = g2'*p;

    ftry    = f2  - oldf - wolfeFtol*oldg*step;
    gtry    = gp2        - wolfeFtol*oldg;
  end
end

%==================================================================
% The search is done.
% Finish with  x1 = the best point found so far.
%==================================================================
step = sbest;

if  improved
  x1 = x2;
elseif  step > 0
  x1 = x + step*p;
end

iExit = inform;

%-------------------------------------------------------------------
% Auxiliary functions:

function [first,improved,done,iExit,numf,alfa,sbest] =     ...
    srchfg ( first,                                        ...
             maxf, numf,                                   ...
             alfmax, epsaf,                                ...
             f0, g0, targetgL, targetgU,                   ...
             ftry, gtry,                                   ...
             tolabs, tolrel, toltny,                       ...
             alfa )

persistent   braktd crampd extrap moved wset nsamea nsameb ...
             a  b alfbst fbest gbest  factor xw fw gw xtry ...
             tolA tolR tolmax

%     ==================================================================
%     srchfg  finds a sequence of improving estimates of a minimizer of
%     the univariate function f(alpha) in the interval (0,alfmax].
%     f(alpha) is a smooth function such that  f(0) = 0  and  f'(0) < 0.
%     srchfg requires both  f(alpha)  and  f'(alpha)  to be evaluated at
%     points in the interval.  Estimates of the minimizer are computed
%     using safeguarded cubic interpolation.
%
%     Reverse communication is used to allow the calling program to
%     evaluate f and f'.  Some of the parameters must be set or tested
%     by the calling program.  The remainder would ordinarily be local
%     variables.
%
%     Input parameters (relevant to the calling program)
%     --------------------------------------------------
%
%     first         must be true on the first entry. It is subsequently
%                   altered by srchfg.
%
%     debug         specifies whether detailed output is wanted.
%
%     maxf          is an upper limit on the number of times srchfg is
%                   to be entered consecutively with done = false
%                   (following an initial entry with first = true).
%
%     alfa          is the first estimate of a minimizer.  alfa is
%                   subsequently altered by srchfg (see below).
%
%     alfmax        is the upper limit of the interval to be searched.
%
%     epsaf         is an estimate of the absolute precision in the
%                   computed value of f(0).
%
%     ftry, gtry    are the values of f, f'  at the new point
%                   alfa = alfbst + xtry.
%
%     g0            is the value of f'(0).  g0 must be negative.
%
%     tolabs,tolrel define a function tol(alfa) = tolrel*alfa + tolabs
%                   such that if f has already been evaluated at alfa,
%                   it will not be evaluated closer than tol(alfa).
%                   These values may be reduced by srchfg.
%
%     targetgL      is the lower bound on the target shifted  f'(alfa).
%                   The search is terminated when f(alfa) le 0 and
%                     targetgL le f'(alfa) le targetgU,
%                   with  targetgL < 0 < targetgU.
%
%     targetgU      is the upper bound on the target shifted f'(alfa).
%
%     toltny        is the smallest value that tolabs is allowed to be
%                   reduced to.
%
%     Output parameters (relevant to the calling program)
%     ---------------------------------------------------
%
%     improved      is true if the previous alfa was the best point so
%                   far.  Any related quantities should be saved by the
%                   calling program (e.g., gradient arrays) before
%                   paying attention to the variable done.
%
%     done = false  means the calling program should evaluate
%                      ftry = f(alfa),  gtry = f'(alfa)
%                   for the new trial alfa, and re-enter srchfg.
%
%     done = true   means that no new alfa was calculated.  The value
%                   of iExit gives the result of the search as follows
%
%                   iExit = 1 means the search has terminated
%                             successfully with alfbst < alfmax.
%
%                   iExit = 2 means the search has terminated
%                             successfully with alfbst = alfmax.
%
%                   iExit = 3 means that the search failed to find a
%                             point of sufficient decrease.
%                             The function is either decreasing at
%                             alfmax or maxf function evaluations
%                             have been exceeded.
%
%                   iExit = 4 means alfmax is so small that a search
%                             should not have been attempted.
%
%                   iExit = 5 is never set by srchfg.
%
%                   iExit = 6 means the search has failed to find a
%                             useful step.  The interval of uncertainty
%                             is [0,b] with b < 2*tolabs. A minimizer
%                             lies very close to alfa = 0, or f'(0) is
%                             not sufficiently accurate.
%
%                   iExit = 7 if no better point could be found after
%                             maxf  function calls.
%
%                   iExit = 8 means the input parameters were bad.
%                             alfmax le toltny  or g0 ge zero.
%                             No function evaluations were made.
%
%     numf          counts the number of times srchfg has been entered
%                   consecutively with done = false (i.e., with a new
%                   function value ftry).
%
%     alfa          is the point at which the next function ftry and
%                   derivative gtry must be computed.
%
%     alfbst        should be accepted by the calling program as the
%                   approximate minimizer, whenever srchfg returns
%                   iExit = 1 or 2 (and possibly 3).
%
%     fbest, gbest  will be the corresponding values of f, f'.
%
%
%     The following parameters retain information between entries
%     -----------------------------------------------------------
%
%     braktd        is false if f and f' have not been evaluated at
%                   the far end of the interval of uncertainty.  In this
%                   case, the point b will be at alfmax + tol(alfmax).
%
%     crampd        is true if alfmax is very small (le tolabs).  If the
%                   search fails, this indicates that a zero step should
%                   be taken.
%
%     extrap        is true if xw lies outside the interval of
%                   uncertainty.  In this case, extra safeguards are
%                   applied to allow for instability in the polynomial
%                   fit.
%
%     moved         is true if a better point has been found, i.e.,
%                   alfbst gt 0.
%
%     wset          records whether a second-best point has been
%                   determined it will always be true when convergence
%                   is tested.
%
%     nsamea        is the number of consecutive times that the
%                   left-hand end point of the interval of uncertainty
%                   has remained the same.
%
%     nsameb        similarly for the right-hand end.
%
%     a, b, alfbst  define the current interval of uncertainty.
%                   A minimizer lies somewhere in the interval
%                   [alfbst + a, alfbst + b].
%
%     alfbst        is the best point so far.  It is always at one end
%                   of the interval of uncertainty.  hence we have
%                   either  a lt 0,  b = 0  or  a = 0,  b gt 0.
%
%     fbest, gbest  are the values of f, f' at the point alfbst.
%
%     factor        controls the rate at which extrapolated estimates
%                   of alfa may expand into the interval of uncertainty.
%                   factor is not used if a minimizer has been bracketed
%                   (i.e., when the variable braktd is true).
%
%     fw, gw        are the values of f, f' at the point alfbst + xw.
%                   they are not defined until wset is true.
%
%     xtry          is the trial point in the shifted interval (a, b).
%
%     xw            is such that  alfbst + xw  is the second-best point.
%                   it is not defined until  wset  is true.
%                   in some cases,  xw  will replace a previous  xw
%                   that has a lower function but has just been excluded
%                   from the interval of uncertainty.
%
%
%     Systems Optimization Laboratory, Stanford University, California.
%     Original version February 1982.  Rev. May 1983.
%     Original f77 version 22-August-1985.
%     14 Sep 1992: Introduced quitI, quitF, etc.
%     22 Nov 1995: Altered criterion for reducing the step below tolabs.
%     17 Jul 1997: Removed saved variables for thread-safe version.
%     19 Apr 2000: QuitF only allowed after a move.
%     28 Nov 2019: Switched to the precise Wolfe conditions.
%     ==================================================================

%     Local variables
%     ===============
%
%     closef     is true if the new function ftry is within epsaf of
%                fbest (up or down).
%
%     found      is true if the sufficient decrease conditions hold at
%                alfbst.
%
%     quitF      is true when  maxf  function calls have been made.
%
%     quitI      is true when the interval of uncertainty is less than
%                2*tol.
%  ---------------------------------------------------------------------

badfun = 0;   quitF  = 0;   quitI  = 0;   improved = 0;

half   = 1/2;

if  first
  %---------------------------------------------------------------
  % First entry.  Initialize various quantities, check input data
  % and prepare to evaluate the function at the initial alfa.
  %---------------------------------------------------------------
  first  = 0;
  alfbst = 0;
  fbest  = f0;
  gbest  = g0;
  badfun = alfmax <= toltny  ||  g0 >= 0;
  done   = badfun;
  moved  = 0;
  tolA   = tolabs;   tolR   = tolrel;

  if  ~done
    braktd = 0;
    crampd = alfmax <= tolA;
    extrap = 0;
    wset   = 0;
    nsamea = 0;        nsameb = 0;

    tolmax = tolA  + tolR*alfmax;
    a      = 0;        b      = alfmax + tolmax;
    factor = 5;
    tol    = tolA;
    xtry   = alfa;
  end
else
  %---------------------------------------------------------------
  % Subsequent entries. The function has just been evaluated at
  % alfa = alfbst + xtry,  giving ftry and gtry.
  %---------------------------------------------------------------

  numf   = numf + 1;  nsamea = nsamea + 1;   nsameb = nsameb + 1;

  if  ~braktd
    tolmax = tolA   + tolR*alfmax;
    b      = alfmax - alfbst + tolmax;
  end

  % See if the new step is better.  If alfa is large enough that
  % ftry can be distinguished numerically from 0,  the function
  % is required to be sufficiently negative.

  closef = abs( ftry - fbest ) <=  epsaf;
  if  closef
    improved =  abs( gtry ) <= abs( gbest );
  else
    improved = ftry < fbest;
  end

  if  improved

    % We seem to have an improvement.  The new point becomes the
    % origin and other points are shifted accordingly.

    fw     = fbest;
    fbest  = ftry;
    gw     = gbest;
    gbest  = gtry;
    alfbst = alfa;
    moved  = 1;

    a      = a    - xtry;
    b      = b    - xtry;
    xw     = 0 - xtry;
    wset   = 1;
    extrap =       xw < 0  &&  gbest < 0  ||  xw > 0  &&  gbest > 0;

    % Decrease the length of the interval of uncertainty.

    if  gtry <= 0
      a      = 0;      nsamea = 0;
    else
      b      = 0;      nsameb = 0;
      braktd = 1;
    end
  else

    % The new function value is not better than the best point so
    % far.  The origin remains unchanged but the new point may
    % qualify as xw.  xtry must be a new bound on the best point.

    if  xtry <= 0
      a      = xtry;      nsamea = 0;
    else
      b      = xtry;      nsameb = 0;
      braktd = 1;
    end

    % If xw has not been set or ftry is better than fw, update the
    % points accordingly.

    if  wset
      setxw = ftry < fw  ||  ~extrap;
    else
      setxw = 1;
    end

    if  setxw
      xw     = xtry;
      fw     = ftry;
      gw     = gtry;
      wset   = 1;
      extrap = 0;
    end
  end

  %---------------------------------------------------------------
  % Check the termination criteria.  wset will always be true.
  %---------------------------------------------------------------
  tol    = tolA   + tolR*alfbst;
  truea  = alfbst + a;     trueb  = alfbst + b;

  found  = targetgL   <= gbest  &&  gbest <= targetgU;
  quitF  = numf       >= maxf   &&  moved;
  quitI  = b - a      <= tol + tol;

  if  quitI  &&  ~moved

    % The interval of uncertainty appears to be small enough,
    % but no better point has been found.  Check that changing
    % alfa by b-a changes f by less than epsaf.

    tol    = tol/10;
    tolA = tol;
    quitI  = tol <= toltny  ||  abs(fw) <= epsaf   &&  gw <= epsaf;
  end

  done  = quitF  ||  quitI  ||  found;

  %---------------------------------------------------------------
  % Proceed with the computation of a trial steplength.
  % The choices are...
  % 1. Parabolic fit using derivatives only, if the f values are
  %    close.
  % 2. Cubic fit for a minimizer, using both f and f'.
  % 3. Damped cubic or parabolic fit if the regular fit appears to
  %    be consistently overestimating the distance to a minimizer.
  % 4. Bisection, geometric bisection, or a step of  tol  if
  %    choices 2 or 3 are unsatisfactory.
  %---------------------------------------------------------------
  if  ~done
    xmidpt = (a + b)/2;
    s      = 0;    q      = 0;

    if   closef
      %---------------------------------------------------------
      % Fit a parabola to the two best gradient values.
      %---------------------------------------------------------
      s      = gbest;
      q      = gbest - gw;
    else
      %---------------------------------------------------------
      % Fit cubic through  fbest  and  fw.
      %---------------------------------------------------------
      fitok  = 1;
      r      = 3*(fbest - fw)/xw + gbest + gw;
      absr   = abs( r );
      s      = sqrt( abs( gbest ) ) * sqrt( abs( gw ) );

      % Compute  q =  the square root of  r*r - gbest*gw.
      % The method avoids unnecessary underflow and overflow.

      if    (gw < 0  &&  gbest > 0) || (gw > 0  &&  gbest < 0)
        scale  = absr + s;
        if  scale == 0
          q  = 0;
        else
          q  = scale*sqrt( (absr/scale)^2 + (s/scale)^2 );
        end
      elseif  absr >= s
        q    = sqrt(absr + s)*sqrt(absr - s);
      else
        fitok = 0;
      end

      if  fitok         % Compute a minimizer of the fitted cubic.
        if  xw < 0
          q = - q;
        end
        s  = gbest -  r - q;    q  = gbest - gw - q - q;
      end
    end
    %------------------------------------------------------------
    % Construct an artificial interval  (artifa, artifb)  in which
    % the new estimate of a minimizer must lie.  Set a default
    % value of xtry that will be used if the polynomial fit fails.
    %------------------------------------------------------------
    artifa = a;    artifb = b;
    if  ~braktd

      % A minimizer has not been bracketed.  Set an artificial
      % upper bound by expanding the interval  xw  by a suitable
      % factor.

      xtry   = - factor*xw;
      artifb =   xtry;
      if  alfbst + xtry < alfmax
        factor = 5*factor;
      end
    elseif  extrap

      % The points are configured for an extrapolation.
      % Set a default value of  xtry  in the interval  (a, b)
      % that will be used if the polynomial fit is rejected.  In
      % the following,  dtry  and  daux  denote the lengths of
      % the intervals  (a, b)  and  (0, xw)  (or  (xw, 0),  if
      % appropriate).  The value of  xtry is the point at which
      % the exponents of  dtry  and  daux  are approximately
      % bisected.

      daux = abs( xw );
      dtry = b - a;
      if  daux >= dtry
        xtry = 5*dtry*(0.1 + dtry/daux)/11;
      else
        xtry = sqrt( daux ) * sqrt( dtry )/2;
      end

      if  xw > 0
        xtry = - xtry;
      end

      % Reset the artificial bounds.  If the point computed by
      % extrapolation is rejected,  xtry will remain at the
      % relevant artificial bound.

      if  xtry <= 0
        artifa = xtry;
      else
        artifb = xtry;
      end
    else

      % The points are configured for an interpolation.  The
      % default value xtry bisects the interval of uncertainty.
      % the artificial interval is just (a, b).

      xtry   = xmidpt;
      if  nsamea >= 3  ||  nsameb >= 3

        % If the interpolation appears to be overestimating the
        % distance to a minimizer,  damp the interpolation.

        factor = factor / 5;
        s      = factor * s;
      else
        factor = 1;
      end
    end
    %------------------------------------------------------------
    % The polynomial fits give  (s/q)*xw  as the new step.
    % Reject this step if it lies outside  (artifa, artifb).
    %------------------------------------------------------------
    if  q ~= 0
      if  q < 0
        s = - s;  q = - q;
      end
      if  s*xw >= q*artifa  &&  s*xw <= q*artifb

        % Accept the polynomial fit.

        if  abs( s*xw ) >= q*tol
          xtry = (s/q)*xw;
        else
          xtry = 0;
        end
      end
    end
  end
end
%==================================================================
if  ~done
  alfa  = alfbst + xtry;
  if  braktd  ||  alfa < alfmax - tolmax

    % The function must not be evaluated too close to a or b.
    % (It has already been evaluated at both those points.)

    if  xtry <= a + tol  ||  xtry >= b - tol
      if  half*(a + b) <= 0
        xtry = - tol;
      else
        xtry =   tol;
      end
      alfa = alfbst + xtry;
    end
  else

    % The step is close to, or larger than alfmax, replace it by
    % alfmax to force evaluation of  f  at the boundary.

    braktd = 1;
    xtry   = alfmax - alfbst;
    alfa   = alfmax;
  end
end

%------------------------------------------------------------------
% Exit.
%------------------------------------------------------------------
sbest = alfbst;
iExit = 0;

if  done
  if     badfun
    iExit = 8;             % bad arguments
  elseif  found
    if  sbest < alfmax
      iExit = 1;           % Sufficient decrease
    else
      iExit = 2;           % Suff. Decrease on the boundary
    end
  elseif  moved
    iExit   = 3;           % Decr at boundary or max funs
  elseif  quitF
    iExit   = 7;           % No new point after max funs
  elseif  crampd
    iExit   = 4;           % alfmax too mall
  else
    iExit   = 6;           % [a,b] too small
  end
end

%-------------------------------------------------------------------
function [tolabs,tolrel,toltny,epsaf] = tols( stepMax,f,p,x )
% tols defines various tolerances for the line search.
%
% Set the input parameters for srchfg.
%
% epsaf   is the absolute function precision. If f(x1) and f(x2) are
%         as close as  epsaf,  we cannot safely conclude from the
%         function values alone which of x1 or x2 is a better point.
%
% tolabs  is an estimate of the absolute spacing between points
%         along  p.  This step should produce a perturbation
%         of  epsaf  in the merit function.
%
% tolrel  is an estimate of the relative spacing between points
%         along  p.
%
% toltny  is the minimum allowable absolute spacing between points
%         along  p.
%

eps0   = eps^(3.5/5); % eps^(4/5)
epsrf  = 10^(-9);

epsaf  = max( epsrf, eps )*(1 + abs(f));
tolax  = eps0;
tolrx  = eps0;

xnorm  = norm(x);
pnorm  = norm(p);

t      = xnorm*tolrx  +  tolax;
if  t < pnorm*stepMax
  tolabs = t/pnorm;
else
  tolabs = stepMax;
end
tolrel = max( tolrx, eps );

t      = max( abs(p)./(abs(x)*tolrx + tolax) );

if  t*tolabs > 1
  toltny = 1 / t;
else
  toltny = tolabs;
end
