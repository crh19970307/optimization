import numpy as np
import scipy 
from numpy.linalg import norm
import sympy

from solver import inexact_Newton,LBFGS
from linesearch import armijogoldstain,wolfe,GLL
from experiment import PenaltyI_function,Trigonometric_function,Extended_Rosenbrock_function,Extended_Powell_singular_function


import json

solver_list = [inexact_Newton,LBFGS]
search_list = ['armijogoldstain','wolfe','GLL']

# debug begin

# debug end

if __name__ == '__main__':
    jsonresult={}

    # PenaltyI_function
    problem='pif'
    jsonresult[problem]={}

    for n in [1000,5000,10000]:
        jsonresult[problem][n]={}
        func = PenaltyI_function(n)
        initpoint=np.arange(1.0,1.0+n)
        
        # IN choice1
        x, f1, g1, niter, neval, duration, x_list, f_list =\
            inexact_Newton(func,initpoint,eps=1e-8,choice=1,maxiter=500)
        jsonresult[problem][n]['IN1']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,n,' IN1 ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # IN choice2
        x, f1, g1, niter, neval, duration, x_list, f_list =\
            inexact_Newton(func,initpoint,eps=1e-8,choice=2,maxiter=500)
        jsonresult[problem][n]['IN2']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,'neval':neval,\
            'duration':duration,'f_list':f_list}
        print(problem,n,' IN2 ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # LBFGS choice1
        for m in [5,9,15]:
            x, f1, g1, niter, neval, duration, x_list, f_list =\
                LBFGS(func,initpoint,eps=1e-8,choice=1,m=m,maxiter=500)
            jsonresult[problem][n]['LBFGS1_m{}'.format(m)]={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,'neval':neval,\
                'duration':duration,'f_list':f_list}
            print(problem,n,' LBFGS1_m{} '.format(m),'fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')
  
        # LBFGS choice2
        for m in [5,9,15]:
            x, f1, g1, niter, neval, duration, x_list, f_list =\
                LBFGS(func,initpoint,eps=1e-8,choice=2,m=m,maxiter=500)
            jsonresult[problem][n]['LBFGS2_m{}'.format(m)]={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,'neval':neval,\
                'duration':duration,'f_list':f_list}
            print(problem,n,' LBFGS2_m{} '.format(m),'fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')
  
    # Trigonometric_function
    problem='tf'
    jsonresult[problem]={}
    
    for n in [1000,5000,10000]:
        jsonresult[problem][n]={}
        func = Trigonometric_function(n)
        initpoint=np.array([1.0/n]*n)
        
        # IN choice1
        x, f1, g1, niter, neval, duration, x_list, f_list =\
            inexact_Newton(func,initpoint,eps=1e-8,choice=1,maxiter=500)
        jsonresult[problem][n]['IN1']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,n,' IN1 ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # IN choice2
        x, f1, g1, niter, neval, duration, x_list, f_list =\
            inexact_Newton(func,initpoint,eps=1e-8,choice=2,maxiter=500)
        jsonresult[problem][n]['IN2']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,'neval':neval,\
            'duration':duration,'f_list':f_list}
        print(problem,n,' IN2 ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # LBFGS choice1
        for m in [5,9,15]:
            x, f1, g1, niter, neval, duration, x_list, f_list =\
                LBFGS(func,initpoint,eps=1e-8,choice=1,m=m,maxiter=500)
            jsonresult[problem][n]['LBFGS1_m{}'.format(m)]={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,'neval':neval,\
                'duration':duration,'f_list':f_list}
            print(problem,n,' LBFGS1_m{} '.format(m),'fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')
  
        # LBFGS choice2
        for m in [5,9,15]:
            x, f1, g1, niter, neval, duration, x_list, f_list =\
                LBFGS(func,initpoint,eps=1e-8,choice=2,m=m,maxiter=500)
            jsonresult[problem][n]['LBFGS2_m{}'.format(m)]={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,'neval':neval,\
                'duration':duration,'f_list':f_list}
            print(problem,n,' LBFGS2_m{} '.format(m),'fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')
    
    # Extended_Rosenbrock_function
    problem='erf'
    jsonresult[problem]={}
    
    for n in [1000,5000,10000]:
        jsonresult[problem][n]={}
        func = Extended_Rosenbrock_function(n)
        initpoint=np.array([-1.2,1]*(n//2))
        
        # IN choice1
        x, f1, g1, niter, neval, duration, x_list, f_list =\
            inexact_Newton(func,initpoint,eps=1e-8,choice=1,maxiter=500)
        jsonresult[problem][n]['IN1']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,n,' IN1 ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # IN choice2
        x, f1, g1, niter, neval, duration, x_list, f_list =\
            inexact_Newton(func,initpoint,eps=1e-8,choice=2,maxiter=500)
        jsonresult[problem][n]['IN2']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,'neval':neval,\
            'duration':duration,'f_list':f_list}
        print(problem,n,' IN2 ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # LBFGS choice1
        for m in [5,9,15]:
            x, f1, g1, niter, neval, duration, x_list, f_list =\
                LBFGS(func,initpoint,eps=1e-8,choice=1,m=m,maxiter=500)
            jsonresult[problem][n]['LBFGS1_m{}'.format(m)]={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,'neval':neval,\
                'duration':duration,'f_list':f_list}
            print(problem,n,' LBFGS1_m{} '.format(m),'fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')
  
        # LBFGS choice2
        for m in [5,9,15]:
            x, f1, g1, niter, neval, duration, x_list, f_list =\
                LBFGS(func,initpoint,eps=1e-8,choice=2,m=m,maxiter=500)
            jsonresult[problem][n]['LBFGS2_m{}'.format(m)]={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,'neval':neval,\
                'duration':duration,'f_list':f_list}
            print(problem,n,' LBFGS2_m{} '.format(m),'fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')
    
    # Extended_Powell_singular_function
    problem='epsf'
    jsonresult[problem]={}
    
    for n in [100,1000,5000,10000]:
        jsonresult[problem][n]={}
        func = Extended_Powell_singular_function(n)
        initpoint=np.array([3,-1,0,1]*(n//4))
        
        # IN choice1
        x, f1, g1, niter, neval, duration, x_list, f_list =\
            inexact_Newton(func,initpoint,eps=1e-8,choice=1,maxiter=500)
        jsonresult[problem][n]['IN1']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,n,' IN1 ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # IN choice2
        x, f1, g1, niter, neval, duration, x_list, f_list =\
            inexact_Newton(func,initpoint,eps=1e-8,choice=2,maxiter=500)
        jsonresult[problem][n]['IN2']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,'neval':neval,\
            'duration':duration,'f_list':f_list}
        print(problem,n,' IN2 ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # LBFGS choice1
        for m in [5,9,15]:
            x, f1, g1, niter, neval, duration, x_list, f_list =\
                LBFGS(func,initpoint,eps=1e-8,choice=1,m=m,maxiter=500)
            jsonresult[problem][n]['LBFGS1_m{}'.format(m)]={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,'neval':neval,\
                'duration':duration,'f_list':f_list}
            print(problem,n,' LBFGS1_m{} '.format(m),'fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')
  
        # LBFGS choice2
        for m in [5,9,15]:
            x, f1, g1, niter, neval, duration, x_list, f_list =\
                LBFGS(func,initpoint,eps=1e-8,choice=2,m=m,maxiter=500)
            jsonresult[problem][n]['LBFGS2_m{}'.format(m)]={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,'neval':neval,\
                'duration':duration,'f_list':f_list}
            print(problem,n,' LBFGS2_m{} '.format(m),'fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')
        
    with open('result.json','w') as f:
        json.dump(jsonresult,f)