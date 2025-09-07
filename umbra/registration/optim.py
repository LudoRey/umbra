import numpy as np
from typing import Callable
from umbra.common.utils import cprint


def get_descent_direction(g, H=None):
    '''
    Compute the descent direction using the Hessian H and the gradient g.
    If H is None, we use the gradient as the descent direction.
    Otherwise, we solve the linear system Hx = -g.
    '''
    if H is None:
        return -g
    else:
        return np.linalg.solve(H, -g)

def line_search_newton(x0: np.ndarray, func: Callable, grad: Callable,
                       hess: Callable = None,
                       c: float = 0.1,
                       delta_max: float | np.ndarray = 1,
                       delta_min: float | np.ndarray = 1e-3, 
                       max_iter: int = 20,
                       callback = lambda x: None):
    '''
    Newton's method with two-way line search based on Armijo rule (see https://en.wikipedia.org/wiki/Backtracking_line_search).
    The hessian is optional; if not provided, will fall back to gradient descent.
    An important variable here describes how much we move at each step : delta := alpha*p, where p is the descent direction.
    The user can define bounds on |delta| to 1) control the step size, and 2) provide a stopping criterion (see below).

    Parameters
    ----------
    - x0 : initial guess
    - func, grad, hess : callables that return the function value, the gradient, and the (optional) hessian.
    - c : positive value to ensure sufficient decrease (see Armijo rule).
    - delta_max : ensures the line search does not make us move by more than delta_max in any direction.
    - delta_min : stop the optimization loop when we move by less than delta_min in all directions.
    '''
    x = x0
    f = func(x0)
    alpha = np.inf # will be set to alpha_max in the first iteration
    # Callback
    callback(0, x, None, f)
    # Early termination flag
    converged = False
    # Optimization loop
    for iter in range(1, max_iter+1):
        # Compute descent direction
        g = grad(x)
        H = hess(x) if hess is not None else None
        p = get_descent_direction(g, H)
        # Armijo condition : decrease must be at least proportional to t
        t = -c*np.dot(g,p) # this should be a positive value
        # 2-way line search
        alpha_max = np.min(delta_max/np.abs(p))
        alpha = np.minimum(alpha_max, alpha) # we reuse last alpha from previous iteration, unless its too big
        f_next = func(x + alpha*p) # proposed value
        accepted = False
        if f - f_next >= alpha*t: # Armijo's condition is initially satisfied: increase the step size until it isn't
            while not accepted:
                alpha *= 2
                if np.any(np.abs(alpha*p) > delta_max): # alpha is too big; stop here
                    alpha /= 2 # Compensate for overshoot
                    accepted = True
                else:
                    f_temp = f_next
                    f_next = func(x + alpha*p)
                    if f - f_next < alpha*t: # Armijo's condition is not satisfied anymore; stop here
                        alpha /= 2 # Compensate for overshoot
                        f_next = f_temp
                        accepted = True
        else: # Armijo's condition is *not* satisfied; reduce step size until it is
            while not (accepted or converged):
                alpha /= 2
                if np.all(np.abs(alpha*p) < delta_min): # Stopping criterion : if not moving by much
                    converged = True
                else:
                    f_next = func(x + alpha*p)
                    if f - f_next >= alpha*t: # Armijo's condition is now satisfied: stop here
                        accepted = True

        if converged:
            cprint("Optimization converged.", color='green')
            return x
        # Accept step
        delta = alpha*p
        x = x + delta
        f = f_next # reuse the value computed during backtracking
        # Callback
        callback(iter, x, delta, f)    
    cprint("Maximum number of iterations reached. Optimization may have not converged.", color='yellow')
    return x
