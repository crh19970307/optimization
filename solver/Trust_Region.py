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

def Trust_Region(fun, x0, eps=1e-8, deltak=1e-1, epsilon=1e-8, maxiter=1000,method='Hebden'):
    """Trust Region method 

    Parameters
    ----------
    fun: object
        objective function, with callable method f, g and G
    x0: ndarray
        initial point
    eps: float, optional
        tolerance, used for convergence criterion
    deltak: float, optional
        Trust region radius
    epsilon: float, optional
        parameter for Inexact Newton  
    maxiter: int, optional
        maximum number of iterations
    method: string, optional
        Trust Region Method, including Hebden, Cauchy Point, DogLeg, 2D Subspace Minimization
        
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
    G = fun.G(x0)
    niter = 0
    neval = 2
    x_list = [x.tolist()]
    f_list = [f1]
    while (abs(f1 - f0) > eps) :
        f1 = fun.f(x)
        g1 = fun.g(x)
        G = fun.G(x)
        if method == 'Hebden':
            niu = 0
            while True:
                try: # test if positive definite
                    L = np.linalg.cholesky(G + niu * np.eye(G.shape[0]))
                    break
                except np.linalg.LinAlgError:
                    a,b=np.linalg.eig(G)
                    niu=-1.01*min(a)
            while(True):
                Ginv = np.linalg.inv(G+niu*np.eye(G.shape[0]))
                d = -Ginv.dot(g1)
                dp = -Ginv.dot(d)
                if (niu == 0 and norm(d)<deltak) or np.abs(norm(d)-deltak)<epsilon * deltak:
                    break
                else:
                    phi = norm(d) - deltak
                    phip = np.dot(d,dp)/norm(d)
                    niu = niu - (phi+deltak)/deltak * phi/phip
        elif method == 'Cauchy':
            tau = 1 if g1.dot(G).dot(g1) <= 0 else min(1,norm(g1)**3/(deltak*g1.dot(G).dot(g1)))
            d = - tau*deltak/norm(g1) * g1
        elif method == 'DogLeg':
            niu = 0
            while True:
                try: # test if positive definite
                    L = np.linalg.cholesky(G + niu * np.eye(G.shape[0]))
                    break
                except np.linalg.LinAlgError:
                    a,b=np.linalg.eig(G)
                    niu=-1.1*min(a)

            G = G + niu * np.eye(G.shape[0])
            Ginv = np.linalg.inv(G)
            dN = -Ginv.dot(g1)
            if norm(dN) <= deltak:
                d = dN
            else:
                dSD = -g1
#                 alpha = inexact_linesearch_armijogoldstain(fun, x, dSD)
                tau = 1 if g1.dot(G).dot(g1) <= 0 else min(1,norm(g1)**3/(deltak*g1.dot(G).dot(g1)))
                alpha = tau*deltak/norm(g1)
                if alpha * norm(dSD) >= deltak:
                    d = deltak * dSD / norm(dSD)
                else:
                    coefficient = [norm(dN)**2 + alpha**2 * norm(dSD)**2 - 2*alpha*dN.dot(dSD), \
                                   2*(alpha*dN.dot(dSD) - alpha**2 * norm(dSD)**2), alpha**2 * norm(dSD)**2 -deltak**2]
                    root = np.roots(coefficient)
                    beta = root[0] if root[0]<=1 and root[0]>=0 else root[1]
                    d = (1-beta)*alpha*dSD + beta*dN
        elif method == '2DSubMin'  :
            niu = 0
            while True:
                try: # test if positive definite
                    L = np.linalg.cholesky(G + niu * np.eye(G.shape[0]))
                    break
                except np.linalg.LinAlgError:
                    a,b=np.linalg.eig(G)
                    niu=-1.1*min(a)
                    
            G = G + niu * np.eye(G.shape[0])
            Ginv = np.linalg.inv(G)
            dN = -Ginv.dot(g1)
            if norm(dN) <= deltak:
                d=dN
            else:
                d1=-g1
                d2=dN
                dx = (d1-d2*d1.dot(d2)/norm(d2)**2 * d2)
                dx=dx/(norm(dx))
                dy = d2/(norm(d2))
                a = 0.5 * dx.dot(G).dot(dx)*deltak**2
                b = 0.5 * dy.dot(G).dot(dy)*deltak**2
                c = dx.dot(G).dot(dy)*deltak**2
                e = g1.dot(dx)*deltak
                f = g1.dot(dy)*deltak
                coefficient = [e-c,2*f - 4*b + 4*a, 6*c, 2*f + 4*b - 4*a,-c-e]
                root = np.roots(coefficient)
                root = np.real(root[np.isreal(root)])
                result= []
                dlist=[]
                for item in root:
                    d = 2*item/(1+item**2) *deltak* dx + (1-item**2)/(1+item**2) *deltak* dy
                    dlist.append(d)
                    result.append(g1.dot(d)+0.5 * d.dot(G).dot(d))
                index = np.argmin(result)
                d = dlist[index]
        else:
            raise NotImplementedError
            
        x = x + d

        f0 = f1
        g0 = g1
        G0 = G
        f1 = fun.f(x)
        g1 = fun.g(x)
        G = fun.G(x)
        
        deltaf = f0 - f1
        deltaq = -g0.dot(d) - 1/2 * d.dot(G0).dot(d)
        gamma = deltaf / deltaq
        
        if gamma<0.25:
            deltak/=4
        elif gamma>0.75 and norm(d) == deltak:
            deltak*=3
        else:
            deltak = deltak

        if gamma<=0:
            x=x-d
        else:
            x=x
        
        f_list.append(f1)       
        niter += 1
        neval += 3
        if niter == maxiter:
            break
    duration = time() - start_time
    return x, f1, g1, niter, neval, duration, x_list, f_list
