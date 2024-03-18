import torch
from functools import reduce
from torch.optim import Optimizer

__all__ = ['COMP']

def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.


def _strong_wolfe(obj_func,
                  x,
                  t,
                  d,
                  f,
                  g,
                  gtd,
                  c1=1e-4,
                  c2=0.9,
                  tolerance_change=1e-9,
                  max_ls=25):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]
    return f_new, g_new, t, ls_func_evals


class COMP(Optimizer):
    """Implements the compact representation algorithm.

    ########################
    #
    # This is a modification of the LBFGS Pytorch implementation from
    # https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html#torch.optim.LBFGS
    #
    # It implements a compact representation, which reduces in a special case to LBFGS 
    #
    ########################
    # 03/07/24, J.B., initial implementation
    # 03/08/24, J.B., preparation of compact formula
    # 03/13/24, J.B., testing limited memory updates


    #
    # The comments below are from the original implementation
    #

    Heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`_.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Args:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-7).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """

    def __init__(self,
                 params,
                 lr=1,
                 lr_1=0.01,
                 max_iter=20,
                 max_eval=None,
                 tolerance_grad=1e-7,
                 tolerance_change=1e-9,
                 history_size=100,
                 line_search_fn=None,
                 whichV="s",
                 whichInit=None,
                 whichG=None,
                 isPD=True,
                 printIt=False):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            lr_1=lr_1,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
            whichV=whichV,
            whichInit=whichInit,
            whichG=whichG,
            isPD=isPD,
            printIt=printIt)
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("COMP doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad


    @torch.no_grad()
    def step(self, closure):
        """Perform a single optimization step.
        This implements the compact representation

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group['lr']
        lr_1 = group['lr_1']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']
        whichV = group['whichV']
        whichInit = group['whichInit']
        whichG = group['whichG']
        isPD = group['isPD']
        printIt = group['printIt']

        # NOTE: COMP has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state['func_evals'] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad

        # optimal condition
        if opt_cond:
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        ro = state.get('ro')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')
        
        # Additional traces
        V = state.get('V')                              # History of vk's
        R1 = state.get('R1')                            # triu(Sk'Yk)
        R2 = state.get('R2')                            # triu(Vk'Yk)
        M2 = state.get('M2')                            # YkYk
        hk = state.get('hk')                            # Gradient or another vec. for the search        
        vk = state.get('vk')
        dk = state.get('dk')                            # diagonal for testing purposes

        b1 = state.get('b1')                            # buffers 1,2,3
        b2 = state.get('b2')                            
        b3 = state.get('b3')                            

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state['n_iter'] += 1

            # print(flat_grad)

            ############################################################
            # compute gradient descent direction
            ############################################################
            if state['n_iter'] == 1:
                d = flat_grad.neg()
                d1 = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1

                V = []                                          # initializing V
                dk = torch.zeros((history_size,1))              # initializing the diagonal

                R1 = torch.zeros((history_size,history_size))   # upper triangular SY
                R2 = torch.zeros((history_size,history_size))   # upper triangular VY
                M2 = torch.zeros((history_size,history_size))   # full YY

                b1 = torch.zeros((history_size,1))              # buffer 1 
                b2 = torch.zeros((history_size,1))              # buffer 2
                b3 = torch.zeros((history_size,1))              # buffer 3

                hk = torch.zeros_like(d)

            else:
                # do comp update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s

                #
                # update vk
                #
                match whichV:
                    case "s":
                        vk = s # s.copy_()
                    case "y":
                        vk = y  #y.copy_()
                    case "g":
                        vk = flat_grad.neg()
                    case "ag":
                        vk = vk.mul(0.9).add(flat_grad.neg().mul(0.1))      # moving average
                
                if (ys > 1e-10) | (isPD == False):
                    # updating memory
                    nl = len(old_dirs)
                    nlm1 = nl-1
                    if nl == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)
                        
                        # Limited memory matrices
                        V.pop(0)
                        R1[0:nlm1,0:nlm1] = R1[1:nl,1:nl]
                        R2[0:nlm1,0:nlm1] = R2[1:nl,1:nl]
                        M2[0:nlm1,0:nlm1] = M2[1:nl,1:nl] 
                        dk[0:nlm1,0]        = dk[1:nl,0].clone()


                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1. / ys)

                    if nl < history_size:
                        nl = nl +1
                        nlm1 = nlm1 + 1

                    # Updates for new containers
                    #dk.append(ys)
                    dk[nlm1] = ys
                    V.append(vk)

                    for i in range(nl):
                        R1[i,nlm1] = old_stps[i].dot(y)
                        R2[i,nlm1] = V[i].dot(y)
                        M2[i,nlm1] = old_dirs[i].dot(y)
                        M2[nlm1,i] = M2[i,nlm1]

                    # if nl > 1:
                    #     M2[nlm1,0:nl] = M2[0:nl,nlm1]

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)


                    ##################### Debugging #####################
                    
                    # if state['n_iter'] > 3:
                    #     print(s)
                        #for i in range(nl):
                            #print(old_stps[i])
                            
                    # For debugging
                    # print(dk[0:nl])
                    # print(R1[0:nl,0:nl])
                    # print(R2[0:nl,0:nl])
                    # print(M2[0:nl,0:nl])

                    # Computing data arrays from scratch (for comparison)
                    # R1C = torch.zeros((history_size,history_size))
                    # R2C = torch.zeros((history_size,history_size))
                    # M2C = torch.zeros((history_size,history_size))
                    # dkC = torch.zeros((history_size,1))

                    # for i in range(nl):
                    #     for j in range(i,nl):
                    #         R1C[i,j] = old_stps[i].dot(old_dirs[j])
                    #         R2C[i,j] = V[i].dot(old_dirs[j])
                    #         M2C[i,j] = old_dirs[i].dot(old_dirs[j])
                    #         M2C[j,i] = M2C[i,j]
                    #     dkC[i] = old_stps[i].dot(old_dirs[i])

                    # if nl > 1:
                    #     M2C[nlm1,0:nl] = M2C[0:nl,nlm1]
                        
                    # Print errors
                    # print(dk[0:nl]-dkC[0:nl])
                    # print(R1[0:nl,0:nl]-R1C[0:nl,0:nl])
                    # print(R2[0:nl,0:nl]-R2C[0:nl,0:nl])
                    # print(M2[0:nl,0:nl]-M2C[0:nl,0:nl])

                    #print(M2C[0:nl,0:nl])

                #
                # Updating the initialization
                #
                match whichInit:
                    case "one":
                        H_diag = 1.0
                    case "lr_1":
                        H_diag = lr_1
                
                num_old = len(old_dirs)
                
                #
                # Two loop recursion (for comparison purposes)
                #
                # # compute the approximate (L-BFGS) inverse Hessian
                # # multiplied by the gradient
                # 

                # if 'al' not in state:
                #     state['al'] = [None] * history_size
                # al = state['al']

                # # iteration in L-BFGS loop collapsed to use just one buffer
                # q = flat_grad.neg()
                # for i in range(num_old - 1, -1, -1):
                #     al[i] = old_stps[i].dot(q) * ro[i]
                #     q.add_(old_dirs[i], alpha=-al[i])

                # # multiply by initial Hessian
                # # r/d is the final direction
                # d = r = torch.mul(q, H_diag)
                # for i in range(num_old):
                #     be_i = old_dirs[i].dot(r) * ro[i]
                #     r.add_(old_stps[i], alpha=al[i] - be_i)


                # # # Debugging
                # # # Testing the secant condition
                # # q = torch.zeros_like(y)
                # # q.copy_(y)
                # # for i in range(num_old - 1, -1, -1):
                # #     al[i] = old_stps[i].dot(q) * ro[i]
                # #     q.add_(old_dirs[i], alpha=-al[i])

                # # # multiply by initial Hessian
                # # # r/d is the final direction
                # # s2 = torch.mul(q, H_diag)
                # # for i in range(num_old):
                # #     be_i = old_dirs[i].dot(s2) * ro[i]
                # #     s2.add_(old_stps[i], alpha=al[i] - be_i)


                #
                # Compact representation
                #
                d = flat_grad.neg()
                if num_old > 0:
                    
                    for i in range(num_old):
                        b1[i] = V[i].dot(d)
                        b2[i] = old_stps[i].dot(d) - H_diag * (old_dirs[i].dot(d))

                    #print(b1[0:num_old])
                    #print(b2[0:num_old])

                    b1[0:num_old] = torch.linalg.solve_triangular(R2[0:num_old,0:num_old],b1[0:num_old],upper=True)
                    b3[0:num_old] = b1[0:num_old]

                    #print(b1[0:num_old])
                    #print(b3[0:num_old])

                    b1[0:num_old] = R1[0:num_old,0:num_old] @ b1[0:num_old] + \
                        R1[0:num_old,0:num_old].T @ b1[0:num_old] - dk[0:num_old].mul(b1[0:num_old]) - \
                        H_diag * (M2[0:num_old,0:num_old] @ b1[0:num_old] )

                    #print(b1[0:num_old])
                    #print(b2[0:num_old]-b1[0:num_old])

                    b1[0:num_old] = torch.linalg.solve_triangular(torch.t(R2[0:num_old,0:num_old]),(b2[0:num_old]-b1[0:num_old]),upper=False)

                    #print(b1[0:num_old])

                    b2[0:num_old] = b3[0:num_old]

                    d.mul_(H_diag)

                    for i in range(num_old):
                        d = d + V[i].mul(b1[i]) + old_stps[i].mul(b2[i]) - old_dirs[i].mul(H_diag*b2[i])

                    #print(d1)

                #
                # Debugging Secant error
                #       
                #s1 = torch.zeros_like(y)
                #s1.copy_(y)                
                #print(y)
                #print(s1)

                # if num_old > 0:
                    
                #     for i in range(num_old):
                #         b1[i] = V[i].dot(s1)
                #         b2[i] = old_stps[i].dot(s1) - H_diag * (old_dirs[i].dot(s1))

                #     b1[0:num_old] = torch.linalg.solve_triangular(R2[0:num_old,0:num_old],b1[0:num_old],upper=True)
                #     b3[0:num_old] = b1[0:num_old]

                #     b1[0:num_old] = R1[0:num_old,0:num_old] @ b1[0:num_old] + \
                #         R1[0:num_old,0:num_old].T @ b1[0:num_old] - dk[0:num_old].mul(b1[0:num_old]) - \
                #         H_diag * (M2[0:num_old,0:num_old] @ b1[0:num_old] )

                #     b1[0:num_old] = torch.linalg.solve_triangular(R2[0:num_old,0:num_old].T,b2[0:num_old]-b1[0:num_old],upper=False)

                #     b2[0:num_old] = b3[0:num_old]

                #     #d1.mul_(H_diag)
                #     s1 = H_diag*s1

                #     for i in range(num_old):
                #         s1 = s1 + V[i].mul(b1[i]) + old_stps[i].mul(b2[i]) - old_dirs[i].mul(H_diag*b2[i])


                #print(s)

                #print(y)

                #print(flat_grad)

                #
                # Print error
                #
                # Comparisons between two loop and compact representation
                # print(float(sum(d-d1)))

                # print(float(sum(s-s1)))

                # print(float(sum(s-s2)))

                # d.copy_(d1)
            
                

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state['n_iter'] == 1:
                 t = min(1., 1. / flat_grad.abs().sum()) * lr
                # Modification of the initial learning rate
                #t = min(1., 1. / flat_grad.abs().sum()) * 1.0
                #t = min(t,lr)

            else:
                t = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # directional derivative is below tolerance
            if gtd > -tolerance_change:
                break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn != "strong_wolfe":
                    raise RuntimeError("only 'strong_wolfe' is supported")
                else:
                    x_init = self._clone_param()

                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d)

                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        obj_func, x_init, t, d, loss, flat_grad, gtd)
                self._add_grad(t, d)
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    with torch.enable_grad():
                        loss = float(closure())
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            # optimal condition
            if opt_cond:
                break

            # lack of progress
            if d.mul(t).abs().max() <= tolerance_change:
                break

            if abs(loss - prev_loss) < tolerance_change:
                break

        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['ro'] = ro
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss

        state['V'] = V
        state['R1'] = R1
        state['R2'] = R2
        state['M2'] = M2
        state['hk'] = hk
        state['vk'] = vk
        state['dk'] = dk
        state['b1'] = b1
        state['b2'] = b2
        state['b3'] = b3
 
        return orig_loss
