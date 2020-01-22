import numpy as np
import scipy 
from numpy.linalg import norm
import sympy

class powell_badly_scaled_function():
    def __init__(self, dim):
        self.dim = dim
        self.symbol_list =[]
        for i in range(dim):
            self.symbol_list.append(sympy.symbols('x{}'.format(i)))

        r1 = 10**4 * self.symbol_list[0] * self.symbol_list[1] - 1
        r2 = sympy.exp(-self.symbol_list[0]) + sympy.exp(-self.symbol_list[1]) - 1.0001
        self.func = r1**2 + r2**2
        
        self.gfunc = []
        self.Gfunc = []
        for i in range(dim):
            self.gfunc.append(sympy.diff(self.func,self.symbol_list[i]))
        for i in range(dim):
            tmp = []
            for j in range(dim):
                tmp.append(sympy.diff(self.gfunc[i],self.symbol_list[j]))
            self.Gfunc.append(tmp)
    def f(self,x):
        dict={}
        for (key,value) in zip(self.symbol_list,x):
            dict[key]=value
        return float(self.func.evalf(subs= dict) )
    def g(self,x):
        dict={}
        for (key,value) in zip(self.symbol_list,x):
            dict[key]=value        
        result = np.zeros([self.dim])
        for i in range(self.dim):
            result[i] = self.gfunc[i].evalf(subs=dict)
        return result
    def G(self,x):
        dict={}
        for (key,value) in zip(self.symbol_list,x):
            dict[key]=value        
        result = np.zeros([self.dim,self.dim])
        for i in range(self.dim):
            for j in range(self.dim):
                result[i][j] = self.Gfunc[i][j].evalf(subs=dict)
        return result
# pbs = powell_badly_scaled_function(2)