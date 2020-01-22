import numpy as np
import scipy 
from numpy.linalg import norm
import sympy

from linesearch import inexact_linesearch_armijogoldstain
from linesearch import inexact_linesearch_wolfe
from linesearch import inexact_linesearch_GLL

import json
from time import time

def LBFGS(fun, x0, eps=1e-5, m=5, maxiter=1000, choice=1, search_method='armijogoldstain'):
    """L-BFGS method 

    Parameters
    ----------
    fun: object
        objective function, with callable method f, g and G
    x0: ndarray
        initial point
    eps: float, optional
        tolerance, used for convergence criterion
    m: int, optional
        parameter for L-BFGS  
    maxiter: int, optional
        maximum number of iterations
    choice: int, optional
        etak choosing method, including method 1 and method 2
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
    s=[]
    y=[]
    while (norm(g1) > eps*max(1,norm(x))) :
        if choice == 1:
            if niter == 0:
                d = -g1
            else:
                alphai =[]
                q = g1
                for i in reversed(range(min(len(s),m))):
                    alphai.insert(0,(1/s[i].dot(y[i])) * s[i].dot(q))
                    q = q - alphai[0]*y[i]

                r = s[-1].dot(y[-1])/y[-1].dot(y[-1]) * q
                for i in range(min(len(s),m)):
                    beta = (1/s[i].dot(y[i]))*y[i].dot(r)
                    r = r + s[i]*(alphai[i]-beta)
                d = -r
        else:
            if niter==0:
                d=-g1
            elif niter==1:
                gammak=s[-1].dot(y[-1])/y[-1].dot(y[-1])
                Yk=np.array(y).T
                Sk=np.array(s).T
                Dk=np.array([[s[0].dot(y[0])]])
                Rk=np.array([[s[0].dot(y[0])]])   
                YkTYk=np.array([[y[0].dot(y[0])]])
                Hkgk = gammak * g1 + np.concatenate([Sk,gammak*Yk],axis=1).dot(
                    np.concatenate(\
                      [np.concatenate([np.linalg.inv(Rk).T.dot(Dk+gammak*YkTYk).dot(np.linalg.inv(Rk)),np.linalg.inv(Rk)],axis=0),
                        np.concatenate([-np.linalg.inv(Rk).T,np.zeros([len(s),len(s)])],axis=0) ]\
                        ,axis=1))\
                        .dot(np.concatenate([Sk.T.dot(g1),gammak*Yk.T.dot(g1)],axis=0))
                d = -Hkgk
            elif niter<=m:
                gammak=s[-1].dot(y[-1])/y[-1].dot(y[-1])
                Yk=np.array(y).T
                Sk=np.array(s).T
                
                tmp=Dk
                Dk=np.zeros([len(s),len(s)])
                Dk[0:len(s)-1,0:len(s)-1]=tmp
                Dk[len(s)-1,len(s)-1]=s[-1].dot(y[-1])
                
                tmp=Rk
                Rk=np.zeros([len(s),len(s)])
                Rk[0:len(s)-1,0:len(s)-1]=tmp
                Rk[:,len(s)-1]=Sk.T.dot(y[-1])
                
                tmp=YkTYk
                YkTYk=np.zeros([len(s),len(s)])
                YkTYk[0:len(s)-1,0:len(s)-1]=tmp
                YkTYk[len(s)-1,:]=y[-1].T.dot(Yk)
                YkTYk[:,len(s)-1]=Yk.T.dot(y[-1])
                
                Hkgk = gammak * g1 + np.concatenate([Sk,gammak*Yk],axis=1).dot(
                    np.concatenate(\
                      [np.concatenate([np.linalg.inv(Rk).T.dot(Dk+gammak*YkTYk).dot(np.linalg.inv(Rk)),-np.linalg.inv(Rk)],axis=0),
                        np.concatenate([-np.linalg.inv(Rk).T,np.zeros([len(s),len(s)])],axis=0) ]\
                        ,axis=1))\
                        .dot(np.concatenate([Sk.T.dot(g1),gammak*Yk.T.dot(g1)],axis=0))
                d = -Hkgk
            else:
                gammak=s[-1].dot(y[-1])/y[-1].dot(y[-1])
                Yk=np.array(y).T
                Sk=np.array(s).T
                
                tmp=Dk
                Dk=np.zeros([len(s),len(s)])
                Dk[0:len(s)-1,0:len(s)-1]=tmp[1::,1::]
                Dk[len(s)-1,len(s)-1]=s[-1].dot(y[-1])
                
                tmp=Rk
                Rk=np.zeros([len(s),len(s)])
                Rk[0:len(s)-1,0:len(s)-1]=tmp[1::,1::]
                Rk[:,len(s)-1]=Sk.T.dot(y[-1])
                
                tmp=YkTYk
                YkTYk=np.zeros([len(s),len(s)])
                YkTYk[0:len(s)-1,0:len(s)-1]=tmp[1::,1::]
                YkTYk[len(s)-1,:]=y[-1].T.dot(Yk)
                YkTYk[:,len(s)-1]=Yk.T.dot(y[-1])
                
                Hkgk = gammak * g1 + np.concatenate([Sk,gammak*Yk],axis=1).dot(
                    np.concatenate(\
                        [np.concatenate([np.linalg.inv(Rk).T.dot(Dk+gammak*YkTYk).dot(np.linalg.inv(Rk)),-np.linalg.inv(Rk)],axis=0),
                        np.concatenate([-np.linalg.inv(Rk).T,np.zeros([len(s),len(s)])],axis=0) ]\
                        ,axis=1))\
                        .dot(np.concatenate([Sk.T.dot(g1),gammak*Yk.T.dot(g1)],axis=0))
                d = -Hkgk
            
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
        g0 = g1
        f1 = fun.f(x)
        g1 = fun.g(x)
        
        s.append(alpha * d)
        y.append(g1-g0)
        if len(s) > m:
            s.pop(0)
            y.pop(0)
        f_list.append(f1) 
        niter += 1
        neval += 3
        if niter == maxiter:
            break
    duration = time() - start_time
    return x, f1, g1, niter, neval, duration, x_list, f_list