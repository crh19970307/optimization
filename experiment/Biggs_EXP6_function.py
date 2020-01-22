import numpy as np
import scipy 
from numpy.linalg import norm
import sympy

class Biggs_EXP6_function():
    def __init__(self, dim, m):
        self.dim = dim
        self.symbol_list =[]
        for i in range(dim):
            self.symbol_list.append(sympy.symbols('x{}'.format(i)))
        
        r=[]
        for i in range(1,1+m):
            r.append(\
               self.symbol_list[2]*sympy.exp(-0.1*i*self.symbol_list[0])       \
                - self.symbol_list[3]*sympy.exp(-0.1*i*self.symbol_list[1])    \
                 + self.symbol_list[5]*sympy.exp(-0.1*i*self.symbol_list[4])   \
                - sympy.exp(-0.1*i) + 5*sympy.exp(-i) - 3*sympy.exp(-0.4*i)    \
                    )

        self.func = r[0]**2
        for i in range(1,m):
            self.func += r[i]**2
        
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
# be6 = Biggs_EXP6_function(6,13)