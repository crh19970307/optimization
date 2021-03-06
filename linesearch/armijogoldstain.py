import numpy as np
import scipy 
from numpy.linalg import norm
import sympy

def inexact_linesearch_armijogoldstain(fun, x, d, a_min=0, a_max=np.inf, row=0.1, t=2 ):
    """Inexact line search: Armijo-Goldstain rule
    Parameters
    ----------
    fun: object
        objective function, with callable method f and g
    x: ndarray
        current point
    d: ndarray
        current search direction
    a_min: float, optional
        lower bound for step length
    a_max: float, optional
        upper bound for step length
    row: float, optional
        parameter for Armijo-Goldstain condition
    t: float, optional
        parameter for Armijo-Goldstain condition

    Returns
    -------
    alpha: float
        step length that satisfies Armijo-Goldstain condition
    """
    alpha = 1.0
    phi0 = fun.f(x)
    phip0 = fun.g(x)
    cnter=0

    while True:
        if cnter>=20:
            return alpha
        phi_alpha = fun.f(x + alpha*d)
        if not phi_alpha <= phi0 + row * alpha * np.dot(phip0, d):
            a_max = alpha
            alpha = (a_min + a_max)/2
            cnter+=1
            continue
        elif not phi_alpha >= phi0 + (1 - row) * alpha * np.dot(phip0, d):
            a_min = alpha
            if a_max < np.inf:
                alpha = (a_min + a_max)/2
            else:
                alpha = t*alpha
            cnter+=1
            continue
        else:
            return alpha