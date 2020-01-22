import numpy as np
import scipy 
from numpy.linalg import norm
import sympy

def modified_cholesky(G , eps=1e-12, delta= 1e-8):
    '''Modified Cholesky Factorization
    
    Parameters
    ----------
    G: ndarray
        matrix to do modified Cholesky factorization
    eps: float, optional
        machine precision
    delta: float, optional
        parameter for Modified Cholesky  
        
    Returns
    -------
    L,D,E,G_mod
    L: ndarray
        L matrix of modified Cholesky factorization
    D: ndarray
        D matrix of modified Cholesky factorization
    E: ndarray
        E matrix of modified Cholesky factorization   
    G_mod: ndarray
        modified Cholesky factorization matrix    
    '''
    assert G.shape[0] == G.shape[1]
    
    n = G.shape[0]
    L = np.zeros((n, n))
    D = np.zeros((n, n))
    E = np.zeros((n, n))
    C = np.zeros((n, n))
    nu = max(np.sqrt(n ** 2 - 1), 1)
    gamma = max(np.abs(G[i][i]) for i in range(n))
    xi = max(np.abs(G[i][j]) for i in range(n) for j in range(n) if i!=j)
    beta_2 = max([gamma, xi / nu])
    for i in range(n):
        C[i][i] = G[i][i]
    j = 0
    while True:
        c_ii = [np.abs(C[i][i]) for i in range(j, n)]
        q = np.argmax(c_ii) + j
        for s in range(j):
            L[j][s] = C[j][s] / D[s][s]
        for i in range(j + 1, n):
            C[i][j] = G[i][j] - sum(L[j][s]*C[i][s] for s in range(j))
        if j == n - 1:
            theta_j = 0
        else:
            theta_j = max(np.abs(C[i][j]) for i in range(j + 1, n))
        D[j][j] = max([delta, np.abs(C[j][j]), theta_j ** 2 / beta_2])
        E[j][j] = D[j][j] - C[j][j]
        if j == n - 1:
            break
        for i in range(j + 1, n):
            C[i][i] -= C[i][j] ** 2 / D[j][j]
        j += 1
    for i in range(n):
        L[i][i] = 1
    return L, D, E , L.dot(D).dot(L.T)