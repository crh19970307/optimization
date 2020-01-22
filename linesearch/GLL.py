import numpy as np
import scipy 
from numpy.linalg import norm
import sympy

def inexact_linesearch_GLL(fun, x, d, que, row=0.1, M=5, sigma=0.5, a=1):
    """Inexact line search: GLL rule
    Parameters
    ----------
    fun: object
        objective function, with callable method f and g
    x: ndarray
        current point
    d: ndarray
        current search direction
    que: list
        value queue
    row: float, optional
        parameter for GLL rule
    M: integer, optional
        parameter for GLL rule
    sigma: float, optional
        parameter for GLL rule
    a: float, optional
        parameter for GLL rule

    Returns
    -------
    alpha: float
        step length that satisfies GLL rule
    que: list
        updated value queue
    """
    
    phi0 = fun.f(x)
    phip0 = fun.g(x)
    if len(que) >= M:
        que.pop(0) 
    que.append(phi0)     
    hk=0
    alpha = sigma**hk *a

    while True:
#         print(x,alpha,d)
        if hk>=10:
            return alpha, que
        phi_alpha = fun.f(x + alpha*d)
        if  phi_alpha <= max(que)+ row * alpha * np.dot(d,phip0):
            return alpha , que
        else:
            hk+=1
            alpha = sigma**hk *a
            continue