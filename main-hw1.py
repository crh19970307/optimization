import numpy as np
import scipy 
from numpy.linalg import norm
import sympy

from solver import damped_Newton,stable_Newton,fletcher_freeman
from linesearch import armijogoldstain,wolfe,GLL
from experiment import powell_badly_scaled_function,Extended_Powell_singular_function,Biggs_EXP6_function

import json

solver_list = [damped_Newton,stable_Newton,fletcher_freeman]
search_list = ['armijogoldstain','wolfe','GLL']

# debug begin

# debug end

if __name__ == '__main__':
    jsonresult={}

    # powell_badly_scaled_function
    pbs = powell_badly_scaled_function(2)
    jsonresult['pbs']={}
    for solver in solver_list:
        jsonresult['pbs'][solver.__name__]={}
        for search_method in search_list:
            x, f1, g1, niter, neval, duration, x_list, f_list = solver(pbs,np.array([0.,1.]),eps=1e-15, delta = 1e-8,search_method=search_method,maxiter=200)
            jsonresult['pbs'][solver.__name__][search_method]={\
                'x':x.tolist(),\
                'f1':f1,\
                'niter':niter,\
                'neval':neval,\
                'x_list':x_list,\
                'f_list':f_list\
                }
            print('pbs ',solver.__name__,' ',search_method,' ',niter)
            


    # Extended_Powell_singular_function
    jsonresult['espf']={}
    for m in [20,40,60,80]:
        epsf = Extended_Powell_singular_function(m)
        jsonresult['espf'][m]={}
        for solver in solver_list:
            jsonresult['espf'][m][solver.__name__]={}
            for search_method in search_list:
                x, f1, g1, niter, neval, duration, x_list, f_list = solver(epsf,np.array([3,-1,0,1]*(m//4)),eps=1e-15, delta = 1e-8,search_method=search_method,maxiter=200)
                jsonresult['espf'][m][solver.__name__][search_method]={\
                'x':x.tolist(),\
                'f1':f1,\
                'niter':niter,\
                'neval':neval,\
                'x_list':x_list,\
                'f_list':f_list\
                }
                print('espf ',m,' ',solver.__name__,' ',search_method,' ',niter)

    # Biggs_EXP6_function
    jsonresult['be6']={}
    for m in [8,9,10,11,12]:
        jsonresult['be6'][m]={}
        be6 = Biggs_EXP6_function(6,m)
        for solver in solver_list:
            jsonresult['be6'][m][solver.__name__]={}
            for search_method in search_list:
                x, f1, g1, niter, neval, duration, x_list, f_list = solver(be6,np.array([1,2,1,1,1,1]),eps=1e-15, delta = 1e-8,search_method=search_method,maxiter=200)
                jsonresult['be6'][m][solver.__name__][search_method]={\
                'x':x.tolist(),\
                'f1':f1,\
                'niter':niter,\
                'neval':neval,\
                'x_list':x_list,\
                'f_list':f_list\
                }   
                print('be6 ',m,' ',solver.__name__,' ',search_method,' ',niter)
    with open('result.json','w') as f:
        json.dump(jsonresult,f)