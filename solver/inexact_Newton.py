import numpy as np
import scipy 
from numpy.linalg import norm
import sympy
from scipy.sparse import linalg
from linesearch import inexact_linesearch_armijogoldstain
from linesearch import inexact_linesearch_wolfe
from linesearch import inexact_linesearch_GLL

import json
from time import time

def inexact_Newton(fun, x0, eps=1e-5, theta_min=0.1, theta_max=0.5, eta=0.9, t=1e-4, gamma=1, alpha=(1+5**0.5)/2, choice=1, maxiter=1000):
    """Inexact Newton's method 

    Parameters
    ----------
    fun: object
        objective function, with callable method f, g and G
    x0: ndarray
        initial point
    eps: float, optional
        tolerance, used for convergence criterion
    theta_min: float, optional
        parameter for Inexact Newton  
    theta_max: float, optional
        parameter for Inexact Newton  
    eta: float, optional
        parameter for Inexact Newton  
    t: float, optional
        parameter for Inexact Newton  
    gamma: float, optional
        parameter for Inexact Newton  
    alpha: float, optional
        parameter for Inexact Newton  
    choice: int, optional
        etak choosing method, including method 1 and method 2
    maxiter: int, optional
        maximum number of iterations
        
    Returns
    -------
    x: ndarray
        optimal point
    f: float
        optimal function value
    g: float
        optimal point gfunction value
    niter: int
        number of iterations
    neval: int 
        number of function evaluations  
    duration: float
        program run time
    x_list: list
        list of x
    f_list: list
        list of f              
    """
    start_time = time()
    x = x0
    f0 = -np.inf
    f1 = fun.f(x0)
    g1 = fun.g(x0)
    niter = 0
    neval = 2
    x_list = [x.tolist()]
    f_list = [f1]
    while (norm(g1) > eps*max(1,norm(x))) :
        if choice == 1:
            etak = 0.5 if niter==0 else max(etak**((1+5**0.5)/2), np.abs((norm(g1)-norm(g0+G.dot(d)))/norm(g0))) if etak**((1+5**0.5)/2)>0.1 else np.abs((norm(g1)-norm(g0+G.dot(d)))/norm(g0))
        elif choice == 2:
            etak = 0.5 if niter==0 else max(gamma*etak**alpha,gamma*(norm(g1)/norm(g0))**alpha)if gamma*etak**alpha>0.1 else gamma*(norm(g1)/norm(g0))**alpha
        G = fun.G(x)
        etak=np.abs(etak)
        etak = min(etak,eta)
        d ,exitcode = linalg.gmres(G,-g1,tol=etak*1e-3, atol=-1.0, restart=20, maxiter=100 )
#         d ,exitcode = scipy.sparse.linalg.gmres(G,(etak-1)*g1, restart=20, maxiter=100 )
#         while(norm(fun.g(x+d)) > (1-t*(1-etak))*norm(g1)):
#             c = norm(g1)
#             b = 2*g1.T.dot(G).dot(d)
#             a = norm(fun.g(x+d)) - b - c
#             theta = fminbound(lambda x: a*x**2 + b*x + c, theta_min, theta_max)
#             d *= theta
#             etak = 1 - theta * (1-etak)
#             neval+=1
#         x = x + d
        if norm(d)==0:
            duration = time() - start_time
            return x, f1, g1, niter, neval, duration, x_list, f_list
        alpha = inexact_linesearch_armijogoldstain(fun, x, d)
        x = x + alpha*d
        f0 = f1
        g0 = g1
        f1 = fun.f(x)
        g1 = fun.g(x)
        f_list.append(f1)       
        niter += 1
        neval += 3
        if niter == maxiter:
            break
    duration = time() - start_time
    return x, f1, g1, niter, neval, duration, x_list, f_list