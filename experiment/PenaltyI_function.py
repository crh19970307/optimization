import numpy as np
import scipy 
from numpy.linalg import norm
import sympy

class PenaltyI_function():
    def __init__(self, dim):
        self.dim = dim
        self.gamma = 1e-5
    def f(self,x):
        assert x.shape[0] == self.dim
        result = self.gamma*sum((x-1)**2)+(sum(x**2)-0.25)**2
        return float(result)
    def g(self,x):
        assert x.shape[0] == self.dim
        s = sum(x**2)
        result = 2*self.gamma*(x-1)+4*(s-0.25)*x
        return result
    def G(self,x):
        assert x.shape[0] == self.dim
        s = sum(x**2)
        result=8*np.outer(x,x)
        diag=2*self.gamma-1+4*s
        result+=np.eye(self.dim)*diag
        return result