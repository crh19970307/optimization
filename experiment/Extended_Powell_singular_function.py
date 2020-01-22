import numpy as np
import scipy 
from numpy.linalg import norm
import sympy

class Extended_Powell_singular_function():
    def __init__(self, dim):
        self.dim = dim
    def f(self,x):
        assert x.shape[0] == self.dim
        result = sum([(x[4*i]+10*x[4*i+1])**2 + 5*(x[4*i+2]-x[4*i+3])**2 +\
                    (x[4*i+1]-2*x[4*i+2])**4 + 10*(x[4*i]-x[4*i+3])**4 for i in range(self.dim//4)])
        return float(result)
    def g(self,x):
        assert x.shape[0] == self.dim
        result = np.zeros([self.dim])
        for i in range(self.dim):
            if i%4==0:
                result[i]=2*(x[i]+10*x[i+1])+40*(x[i]-x[i+3])**3
            elif i%4==1:
                result[i]=20*(x[i-1]+10*x[i])+4*(x[i]-2*x[i+1])**3
            elif i%4==2:
                result[i]=10*(x[i]-x[i+1])+8*(2*x[i]-x[i-1])**3
            else:
                result[i]=10*(x[i]-x[i-1])+40*(x[i]-x[i-3])**3
        return result
    def G(self,x):
        assert x.shape[0] == self.dim
        result = np.zeros([self.dim,self.dim])
        for i in range(self.dim):
            if i%4 == 0:
                result[i,i]=2+120*(x[i]-x[i+3])**2
                result[i,i+1]=20
                result[i,i+3]=-120*(x[i]-x[i+3])**2
            elif i%4==1:
                result[i,i]=200+12*(x[i]-2*x[i+1])**2
                result[i,i-1]=20 
                result[i,i+1]=-24*(x[i]-2*x[i+1])**2
            elif i%4==2:
                result[i,i]=10+48*(2*x[i]-x[i-1])**2
                result[i,i-1]=-24*(2*x[i]-x[i-1])**2
                result[i,i+1]=-10
            else:
                result[i,i]=10+120*(x[i]-x[i-3])**2
                result[i,i-1]=-10
                result[i,i-3]=-120*(x[i]-x[i-3])**2
        return result

class Extended_Powell_singular_function_sympy():
    def __init__(self, dim):
        # m = dim = 4 * k 
        m = dim
        self.dim = dim
        self.symbol_list =[]
        for i in range(dim):
            self.symbol_list.append(sympy.symbols('x{}'.format(i)))
        r_list = []
        for i in range(m//4):
            r1 = self.symbol_list[4*i+0]+10*self.symbol_list[4*i+1]
            r2 = 5**0.5 *(self.symbol_list[4*i+2]-self.symbol_list[4*i+3])
            r3 = (self.symbol_list[4*i+1]-2*self.symbol_list[4*i+2])**2
            r4 = 10**0.5 *(self.symbol_list[4*i+0]-self.symbol_list[4*i+3])**2
            r_list.append(r1)
            r_list.append(r2)
            r_list.append(r3)
            r_list.append(r4)
        self.func = r_list[0]**2
        for i in range(1,m):
            self.func+=r_list[i]**2
        
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
# epsf = Extended_Powell_singular_function(20)