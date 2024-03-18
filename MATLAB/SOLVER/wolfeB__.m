function [alp,nf,ng,ex] = wolfeB__(x,p,g,f,alp0,c1,c2,func,grad,alpmax,...
                                        stpmin,stpinc,maxit)
%wolfeB Implementation of a strong Wolfe Line-Search
%   [alp,nf,ng,ex] = strongWolfe(alp0,func,grad,alpmax,stpmin,stepinc)
%   
%   Computes the step-length alp for a deterministic optimization algorithm
%   x:      Current iterate,
%   p:      Current update,
%   g:      Current gradient,
%   f:      Current function value,
%   alp0:   initial step, 
%   c1:     Armijo condition (c1=1e-4),
%   c2:     Curvature condition (c2=0.9), 
%   func:   Function handle,
%   grad:   Gradient handle,
%   alpmax: Max step length,
%   stpmin: Minimum step interval,
%   stpinc: Factor to increase step
%   maxit:  Max iterations
%
% The Wolfe conditions are enforced for the "shifted" function
%
%   l(alp) = f(x + alp*p) - (f + c1*alp*p'*g) 
%
%--------------------------------------------------------------------------
% 02/05/24, J.B., initial implementation
% 02/11/24, J.B., name change

%
% Initializations
%
pg      = p'*g;
j       = 0;
nf      = 0;
ng      = 0;
sig     = 1;

an      = alp0;
al      = 0;
ah      = al;
ab      = min(alpmax,an);
stage1  = true;

omega   = @(alp)( func(x+alp*p) - ( f + c1*alp*pg ) );
omegap  = @(alp)( grad(x+alp*p)'*p - c1*pg );

ol      = omega(al);
opl     = omega(al);
oh      = ol;
oph     = opl;

nf      = nf + 1;
ng      = ng + 1;

%
% Main loop
%

while (abs(ah-al) > stpmin || j == 0) && j < maxit

    on  = omega(an);            nf = nf + 1;
    opn = omegap(an);           ng = ng + 1;

    %
    % Start checks
    %
    if on < ol 
        if stage1 == true
            ah = al;
            oh = ol;
            oph = opl;
            al  = an;
            ol  = on;
            opl = opn;
            if opl >= 0
                ab = ah; stage1 = false;
            end
        else
            if opn*(al-an) < 0
                ah = al;
                oh = ol;
                oph = opl;
                ab  = ah;
            end
            al = an;
            ol = on;
            opl = opn;
        end
    else
        if stage1 == true; stage1 = false; end
        ah  = an;
        oh  = on;
        oph = opn;
        ab  = ah;
    end

    %
    % Termination check
    %
    if (ol < 0 && ( abs(opl + c1*pg) <= c2*abs(pg) )) || al > alpmax
        break;
    end

    if stage1 == true
        an  = ab;
        ab  = min(alpmax,sig*al);
        sig = stpinc*sig;
    else

        % Compute at ("alp trial") by interpolating al and ah

        if al < ah
            a1_     = al;
            a2      = ah;
            h1_     =  ol +  ( f + c1*al*pg );
            hp1_    = opl + c1*pg;
            h2      =  oh +  ( f + c1*ah*pg );            
        else
            a1_     = ah;
            a2      = al;
            h1_     =  oh +  ( f + c1*ah*pg );
            hp1_    = oph + c1*pg;
            h2      =  ol +  ( f + c1*al*pg );
        end

        a   = -2*( ( h1_-h2 )/(a1_-a2)^2 - hp1_/(a1_-a2));
        
        at  = - hp1_/a + a1_;

        % Check the bounds
        if al < ab
            ab1 = al;
            ab2 = ab;
        else
            ab1 = ab;
            ab2 = al;
        end

        if at <= ab2 && at >= ab1

            an = at;
        else
            an = 0.5*(al+ah);
        end

    end

    j = j + 1;

end

%
% Termination information 
%

alp = al;
if abs(ah-al) <= stpmin
    ex = -1;
elseif j == maxit
    ex = -2;
elseif al >= alpmax
    ex = 2;
else 
    ex = 1;
end

end



