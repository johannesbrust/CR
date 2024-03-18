      function [x,f,g,stp,info,nfev] ...
       = cvsrch_INTF1(func,n,x,f,g,s,stp,ftol,gtol,xtol, ...
                 stpmin,stpmax,maxfev)
%   Translation of minpack subroutine cvsrch
%   Dianne O'Leary   July 1991
%
%   MODIFICATION: cvsrch_INTF uses two inputs for the function and gradient
%   evaluations, different from the original method, which only used one.
%   12/11/19, J.B.
%
%   MODIFICATION 1: revert back to use only one function to evaluate 
%   the gradient and function values
%   02/21/24, J.B.
%
%     **********
%
%     Subroutine cvsrch
%
%     The purpose of cvsrch is to find a step which satisfies 
%     a sufficient decrease condition and a curvature condition.
%     The user must provide a subroutine which calculates the
%     function and the gradient.
%
%     At each stage the subroutine updates an interval of
%     uncertainty with endpoints stx and sty. The interval of
%     uncertainty is initially chosen so that it contains a 
%     minimizer of the modified function
%
%          f(x+stp*s) - f(x) - ftol*stp*(gradf(x)'s).
%
%     If a step is obtained for which the modified function 
%     has a nonpositive function value and nonnegative derivative, 
%     then the interval of uncertainty is chosen so that it 
%     contains a minimizer of f(x+stp*s).
%
%     The algorithm is designed to find a step which satisfies 
%     the sufficient decrease condition 
%
%           f(x+stp*s) <= f(x) + ftol*stp*(gradf(x)'s),
%
%     and the curvature condition
%
%           abs(gradf(x+stp*s)'s)) <= gtol*abs(gradf(x)'s).
%
%     If ftol is less than gtol and if, for example, the function
%     is bounded below, then there is always a step which satisfies
%     both conditions. If no step can be found which satisfies both
%     conditions, then the algorithm usually stops when rounding
%     errors prevent further progress. In this case stp only 
%     satisfies the sufficient decrease condition.
%
%     The subroutine statement is
%
%        subroutine cvsrch(fcn,n,x,f,g,s,stp,ftol,gtol,xtol,
%                          stpmin,stpmax,maxfev,info,nfev,wa)
%     where
%
%	fcn is the name of the user-supplied subroutine which
%         calculates the function and the gradient.  fcn must 
%      	  be declared in an external statement in the user 
%         calling program, and should be written as follows.
%
%         function [f,g] = fcn(n,x) (Matlab)     (10/2010 change in documentation)
%	  (derived from Fortran subroutine fcn(n,x,f,g) )
%         integer n
%         f
%         x(n),g(n)
%	  ----------
%         Calculate the function at x and
%         return this value in the variable f.
%         Calculate the gradient at x and
%         return this vector in g.
%	  ----------
%	  return
%	  end
%
%       n is a positive integer input variable set to the number
%	  of variables.
%
%	x is an array of length n. On input it must contain the
%	  base point for the line search. On output it contains 
%         x + stp*s.
%
%	f is a variable. On input it must contain the value of f
%         at x. On output it contains the value of f at x + stp*s.
%
%	g is an array of length n. On input it must contain the
%         gradient of f at x. On output it contains the gradient
%         of f at x + stp*s.
%
%	s is an input array of length n which specifies the
%         search direction.
%
%	stp is a nonnegative variable. On input stp contains an
%         initial estimate of a satisfactory step. On output
%         stp contains the final estimate.
%
%       ftol and gtol are nonnegative input variables. Termination
%         occurs when the sufficient decrease condition and the
%         directional derivative condition are satisfied.
%
%	xtol is a nonnegative input variable. Termination occurs
%         when the relative width of the interval of uncertainty 
%	  is at most xtol.
%
%	stpmin and stpmax are nonnegative input variables which 
%	  specify lower and upper bounds for the step.
%
%	maxfev is a positive integer input variable. Termination
%         occurs when the number of calls to fcn is at least
%         maxfev by the end of an iteration.
%
%	info is an integer output variable set as follows:
%	  
%	  info = 0  Improper input parameters.
%
%	  info = 1  The sufficient decrease condition and the
%                   directional derivative condition hold.
%
%	  info = 2  Relative width of the interval of uncertainty
%		    is at most xtol.
%
%	  info = 3  Number of calls to fcn has reached maxfev.
%
%	  info = 4  The step is at the lower bound stpmin.
%
%	  info = 5  The step is at the upper bound stpmax.
%
%	  info = 6  Rounding errors prevent further progress.
%                   There may not be a step which satisfies the
%                   sufficient decrease and curvature conditions.
%                   Tolerances may be too small.
%
%       nfev is an integer output variable set to the number of
%         calls to fcn.
%
%	wa is a work array of length n.
%
%     Subprograms called
%
%	user-supplied......fcn
%
%	MINPACK-supplied...cstep
%
%	FORTRAN-supplied...abs,max,min
%	  
%     Argonne National Laboratory. MINPACK Project. June 1983
%     Jorge J. More', David J. Thuente
%
%     **********
      p5 = .5;
      p66 = .66;
      xtrapf = 4;
      info = 0;
      infoc = 1;

%
%     Check the input parameters for errors.
%
      if (n <= 0 | stp <= 0.0 | ftol < 0.0 |  ...
          gtol < 0.0 | xtol < 0.0 | stpmin < 0.0  ...
          | stpmax < stpmin | maxfev <= 0) 
         return
      end
%
%     Compute the initial gradient in the search direction
%     and check that s is a descent direction.
%
      dginit = g'*s;
      if (dginit >= 0.0) 
          return
      end
%
%     Initialize local variables.
%
      brackt = 0;
      stage1 = 1;
      nfev = 0;
      finit = f;
      dgtest = ftol*dginit;
      width = stpmax - stpmin;
      width1 = 2*width;
      wa = x;
%
%     The variables stx, fx, dgx contain the values of the step, 
%     function, and directional derivative at the best step.
%     The variables sty, fy, dgy contain the value of the step,
%     function, and derivative at the other endpoint of
%     the interval of uncertainty.
%     The variables stp, f, dg contain the values of the step,
%     function, and derivative at the current step.
%
      stx = 0.0;
      fx = finit;
      dgx = dginit;
      sty = 0.0;
      fy = finit;
      dgy = dginit;
%
%     Start of iteration.
%
   while (1)   
%
%        Set the minimum and maximum steps to correspond
%        to the present interval of uncertainty.
%
         if (brackt) 
            stmin = min(stx,sty);
            stmax = max(stx,sty);
         else
            stmin = stx;
            stmax = stp + xtrapf*(stp - stx);
         end 
%
%        Force the step to be within the bounds stpmax and stpmin.
%
         stp = max(stp,stpmin);
         stp = min(stp,stpmax);
%
%        If an unusual termination is to occur then let 
%        stp be the lowest point obtained so far.
%
         if ((brackt & (stp <= stmin | stp >= stmax)) ...
            | nfev >= maxfev-1 | infoc == 0 ...
            | (brackt & stmax-stmin <= xtol*stmax)) 
            stp = stx;
         end
%
%        Evaluate the function and gradient at stp
%        and compute the directional derivative.
%
         x = wa + stp * s;
         %[f,g] = feval(fcn,n,x);
         % f = func(x);
         % g = grad(x);

         [f,g] = func(x);

         nfev = nfev + 1;
         dg = g' * s;
         ftest1 = finit + stp*dgtest;
%
%        Test for convergence.
%
         if ((brackt & (stp <= stmin | stp >= stmax)) | infoc == 0) 
                  info = 6;
         end
         if (stp == stpmax & f <= ftest1 & dg <= dgtest) 
                  info = 5;
         end
         if (stp == stpmin & (f > ftest1 | dg >= dgtest)) 
                  info = 4;
         end
         if (nfev >= maxfev) 
                  info = 3;
         end
         if (brackt & stmax-stmin <= xtol*stmax) 
                  info = 2;
         end
         if (f <= ftest1 & abs(dg) <= gtol*(-dginit)) 
                  info = 1;
         end
%
%        Check for termination.
%
         if (info ~= 0) 
                  return
         end
%
%        In the first stage we seek a step for which the modified
%        function has a nonpositive value and nonnegative derivative.
%
         if (stage1 & f <= ftest1 & dg >= min(ftol,gtol)*dginit) 
                stage1 = 0;
         end
%
%        A modified function is used to predict the step only if
%        we have not obtained a step for which the modified
%        function has a nonpositive function value and nonnegative 
%        derivative, and if a lower function value has been  
%        obtained but the decrease is not sufficient.
%
         if (stage1 & f <= fx & f > ftest1) 
%
%           Define the modified function and derivative values.
%
            fm = f - stp*dgtest;
            fxm = fx - stx*dgtest;
            fym = fy - sty*dgtest;
            dgm = dg - dgtest;
            dgxm = dgx - dgtest;
            dgym = dgy - dgtest;
% 
%           Call cstep to update the interval of uncertainty 
%           and to compute the new step.
%
            [stx,fxm,dgxm,sty,fym,dgym,stp,fm,dgm,brackt,infoc] ...
             = cstep(stx,fxm,dgxm,sty,fym,dgym,stp,fm,dgm, ...
                     brackt,stmin,stmax);
%
%           Reset the function and gradient values for f.
%
            fx = fxm + stx*dgtest;
            fy = fym + sty*dgtest;
            dgx = dgxm + dgtest;
            dgy = dgym + dgtest;
         else
% 
%           Call cstep to update the interval of uncertainty 
%           and to compute the new step.
%
            [stx,fx,dgx,sty,fy,dgy,stp,f,dg,brackt,infoc] ...
             = cstep(stx,fx,dgx,sty,fy,dgy,stp,f,dg, ...
                     brackt,stmin,stmax);
         end
%
%        Force a sufficient decrease in the size of the
%        interval of uncertainty.
%
         if (brackt) 
            if (abs(sty-stx) >= p66*width1) 
              stp = stx + p5*(sty - stx);
            end
            width1 = width;
            width = abs(sty-stx);
         end
%
%        End of iteration.
%
     end
%
%     Last card of subroutine cvsrch.
%


