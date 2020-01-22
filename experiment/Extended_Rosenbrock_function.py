import numpy as np
import scipy 
from numpy.linalg import norm
import sympy

class Extended_Rosenbrock_function():
    def __init__(self, dim):
        self.dim = dim
    def f(self,x):
        assert x.shape[0] == self.dim
        result = sum([(1-x[2*i])**2 + 100*(x[2*i+1]-x[2*i]**2)**2 for i in range(self.dim//2)])
        return float(result)
    def g(self,x):
        assert x.shape[0] == self.dim
        result = np.zeros([self.dim])
        for i in range(self.dim):
            if i%2==0:
                result[i]=2*(x[i]-1)+400*(x[i]**2-x[i+1])*x[i]
            else:
                result[i]=200*(x[i]-x[i-1]**2)
        return result
    def G(self,x):
        assert x.shape[0] == self.dim
        result = np.zeros([self.dim,self.dim])
        for i in range(self.dim):
            if i%2 == 0:
                result[i,i]=2+1200*x[i]**2-400*x[i+1]
                result[i,i+1]=-400*x[i]
            else:
                result[i,i]=200
                result[i,i-1]=-400*x[i-1]
        return result