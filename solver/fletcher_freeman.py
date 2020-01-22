import numpy as np
import scipy 
from numpy.linalg import norm
import sympy

from decomposition import BPdecomposition
from linesearch import inexact_linesearch_armijogoldstain
from linesearch import inexact_linesearch_wolfe
from linesearch import inexact_linesearch_GLL

import json
from time import time

def fletcher_freeman(fun, x0, eps=1e-8, delta=1e-8, maxiter=1000, search_method='armijogoldstain'):
    """Fletcher Freeman Method

    Parameters
    ----------
    fun: object
        objective function, with callable method f, g and G
    x0: ndarray
        initial point
    eps: float, optional
        tolerance, used for convergence criterion
    delta: float, optional
        parameter for fletcher freeman   
    maxiter: int, op8tional
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
        n = G.shape[0]
        L,D,P,y,G_new,L_list,D_list= BPdecomposition(G)
        id = 0
        eigvalue_list ,_= np.linalg.eig(D)
        if norm(G_new)==0:
            d = -g1
        elif (min(eigvalue_list)>0):
            d = -np.linalg.inv(G).dot(g1)
            # 注意这里是G而不是G_new
        elif min(eigvalue_list)<0:
            #todo
            a = np.zeros([n])
            id = 0
            for item in D_list:
                if item.shape[1]==1:
                    a[id]=1 if item[0,0]<0 else 0
                    id+=1
                else:
                    eigvalue, eigvector = np.linalg.eig(item)
                    a[id:id+2] = eigvector[:,0]/norm(eigvector[:,0]) if eigvalue[0]<0 else eigvector[:,1]/eigvector[:,1]  
                    id+=2
            d = np.linalg.solve(L.T, a)  
            d = d if d.dot(g1) <=0 else -d
        else:
            #todo7
            Dknew = np.zeros([n,n])
            Dknewinv = np.zeros([n,n])
            id = 0
            for item in D_list:
                if item.shape[1]==1:
                    Dknew[id,id]=item[0,0] if item[0,0]>0 else 0
                    Dknewinv[id,id]=1/item[0,0] if item[0,0]>0 else 0
                    id+=1
                else:
                    eigvalue, eigvector = np.linalg.eig(item)
                    Dknew[id:id+2,id:id+2] = eigvalue[0]*eigvector[:,0].dot(eigvector[:,0].T) if eigvalue[0]>0 else \
                                    1/eigvalue[1]*eigvector[:,1].dot(eigvector[:,1].T)  
                    Dknewinv[id:id+2,id:id+2] = 1/eigvalue[0]*eigvector[:,0].dot(eigvector[:,0].T) if eigvalue[0]>0 else \
                                    1/eigvalue[1]*eigvector[:,1].dot(eigvector[:,1].T)    
                    id+=2            
            d = -np.linalg.inv(L.T).dot(Dknewinv).dot(np.linalg.inv(L)).dot(g1)
            
        if search_method == 'armijogoldstain':
            alpha = inexact_linesearch_armijogoldstain(fun, x, d)
        elif search_method == 'wolfe':
            alpha = inexact_linesearch_wolfe(fun, x, d)
        elif search_method == 'GLL':
            alpha,que = inexact_linesearch_GLL(fun, x, d,que)
        else:
            print('search method not implemented!')
            return 
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