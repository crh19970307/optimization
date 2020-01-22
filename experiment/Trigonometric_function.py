import numpy as np
import scipy 
from numpy.linalg import norm
import sympy

class Trigonometric_function():
    def __init__(self, dim):
        self.dim = dim
    def f(self,x):
        assert x.shape[0] == self.dim
        s = sum(np.cos(x))
        result = sum((self.dim-s+np.arange(1,self.dim+1)*(1-np.cos(x))-np.sin(x))**2)
        return float(result)
    def g(self,x):
        assert x.shape[0] == self.dim
        s = sum(np.cos(x))
        s2=sum(2*(self.dim-s+np.arange(1,self.dim+1)*(1-np.cos(x))-np.sin(x)))
        result=2*(self.dim-s+np.arange(1,self.dim+1)*(1-np.cos(x))-np.sin(x))*\
        (np.arange(1,self.dim+1)*np.sin(x)-np.cos(x))+s2*np.sin(x)
        return result
    def G(self,x):
        assert x.shape[0] == self.dim
        s = sum(np.cos(x))
        result = 2*np.outer((np.arange(1,self.dim+1)*np.sin(x)-np.cos(x)),np.sin(x))\
                +2*np.outer(self.dim*np.sin(x),np.sin(x))\
                +2*np.outer(np.sin(x),(np.arange(1,self.dim+1)*np.sin(x)-np.cos(x)))
        s2=sum(self.dim-s+np.arange(1,self.dim+1)*(1-np.cos(x))-np.sin(x))
        diag=2*(np.arange(1,self.dim+1)*np.sin(x)-np.cos(x))**2 +\
            2*(self.dim-s+np.arange(1,self.dim+1)*(1-np.cos(x))-np.sin(x))*\
            (np.arange(1,self.dim+1)*np.cos(x)+np.sin(x))+2*s2*np.cos(x)
        result+=np.diag(diag)
        return result