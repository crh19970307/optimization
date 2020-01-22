import numpy as np
import scipy 
from numpy.linalg import norm
import sympy

from decomposition import modified_cholesky
from linesearch import inexact_linesearch_armijogoldstain
from linesearch import inexact_linesearch_wolfe
from linesearch import inexact_linesearch_GLL

import json
from time import time

def stable_Newton(fun, x0, eps=1e-8, delta=1e-8, maxiter=1000, search_method='armijogoldstain'):
    """Stable Newton's method base on Cholesky Factorization

    Parameters
    ----------
    fun: object
        objective function, with callable method f, g and G
    x0: ndarray
        initial point
    eps: float, optional
        tolerance, used for convergence criterion
    delta: float, optional
        parameter for Stable Newton   
    maxiter: int, optional
        maximum number of iterations
    search_method: string, optional
        linesearch method

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
    que = []
    while (abs(f1 - f0) > eps) :
        G = fun.G(x)
        L,D,E,G_mod= modified_cholesky(G)
        if abs(norm(g1))>delta:
            d = np.linalg.solve(G_mod, -g1)
        else:
            argmin = np.argmin(np.array([D[i][i]-E[i][i] for i in range(D.shape[0])]))
            if D[argmin][argmin]>=0:
                duration = time() - start_time
                return x, f1, g1, niter, neval, duration, x_list, f_list

            else:
                et = np.zeros([D.shape[0]])
                et[argmin]=1
                d = np.linalg.solve(L.T, et)
                d = d if d.dot(g1) <=0 else -d
        if search_method == 'armijogoldstain':
            alpha = inexact_linesearch_armijogoldstain(fun, x, d)
        elif search_method == 'wolfe':
            alpha = inexact_linesearch_wolfe(fun, x, d)
        elif search_method == 'GLL':
            alpha,que = inexact_linesearch_GLL(fun, x, d,que)
        else:
            print('search method not implemented!')
            return 
#         alpha,que = inexact_linesearch_GLL(fun, x, d,que)
#         alpha = inexact_linesearch_wolfe(fun, x, d)
#         alpha = inexact_linesearch_armijogoldstain(fun, x, d)
        x = x + alpha * d
        x_list.append(x.tolist())
        f0 = f1
        f1 = fun.f(x)
        g1 = fun.g(x)
        f_list.append(f1)       
        niter += 1
        neval += 3
        if niter == maxiter:
            break

    duration = time() - start_time
    return x, f1, g1, niter, neval, duration, x_list, f_list