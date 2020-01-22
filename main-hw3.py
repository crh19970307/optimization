import numpy as np
import scipy 
from numpy.linalg import norm
import sympy

from solver import damped_Newton,stable_Newton,fletcher_freeman,Trust_Region
from linesearch import armijogoldstain,wolfe,GLL
from experiment import powell_badly_scaled_function,Extended_Powell_singular_function,Biggs_EXP6_function

import json


if __name__ == '__main__':
    jsonresult={}

    # powell_badly_scaled_function
    problem='pbsf'
    jsonresult[problem]={}

    for m in [2]:
        jsonresult[problem][m]={}
        func = powell_badly_scaled_function(m)
        initpoint=np.array([0.,1.])

        # damped_Newton
        x, f1, g1, niter, neval, duration, x_list, f_list = damped_Newton\
            (func,initpoint,eps=1e-15, delta = 1e-8,search_method='armijogoldstain',maxiter=200)
        jsonresult[problem][m]['dN']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,m,' dN ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # stable_Newton
        x, f1, g1, niter, neval, duration, x_list, f_list = stable_Newton\
            (func,initpoint,eps=1e-15, delta = 1e-8,search_method='armijogoldstain',maxiter=200)
        jsonresult[problem][m]['sN']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,m,' sN ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # fletcher_freeman
        x, f1, g1, niter, neval, duration, x_list, f_list = fletcher_freeman\
            (func,initpoint,eps=1e-15, delta = 1e-8,search_method='armijogoldstain',maxiter=200)
        jsonresult[problem][m]['ff']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,m,' ff ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # Hedben
        x, f1, g1, niter, neval,duration, x_list, f_list = Trust_Region\
            (func,initpoint,eps=1e-15, deltak=1e-1, epsilon=1e-8, maxiter=200,method = 'Hebden')
        jsonresult[problem][m]['H']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,m,' H ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # Cauchy
        x, f1, g1, niter, neval,duration, x_list, f_list = Trust_Region\
            (func,initpoint,eps=1e-15, deltak=1e-1, epsilon=1e-8, maxiter=1000,method = 'Cauchy')
        jsonresult[problem][m]['C']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,m,' C ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # DogLeg
        x, f1, g1, niter, neval,duration, x_list, f_list = Trust_Region\
            (func,initpoint,eps=1e-15, deltak=1e-1, epsilon=1e-8, maxiter=200,method = 'DogLeg')
        jsonresult[problem][m]['DL']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,m,' DL ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # 2DSubMin
        x, f1, g1, niter, neval,duration, x_list, f_list = Trust_Region\
            (func,initpoint,eps=1e-15, deltak=1e-1, epsilon=1e-8, maxiter=200,method = '2DSubMin')
        jsonresult[problem][m]['2DSM']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,m,' 2DSM ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')


    # Extended_Powell_singular_function
    problem='epsf'
    jsonresult[problem]={}

    for m in [20,40,60,80]:
        jsonresult[problem][m]={}
        func = Extended_Powell_singular_function(m)
        initpoint=np.array([3,-1,0,1]*(m//4))

        # damped_Newton
        x, f1, g1, niter, neval, duration, x_list, f_list = damped_Newton\
            (func,initpoint,eps=1e-15, delta = 1e-8,search_method='armijogoldstain',maxiter=200)
        jsonresult[problem][m]['dN']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,m,' dN ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # stable_Newton
        x, f1, g1, niter, neval, duration, x_list, f_list = stable_Newton\
            (func,initpoint,eps=1e-15, delta = 1e-8,search_method='armijogoldstain',maxiter=200)
        jsonresult[problem][m]['sN']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,m,' sN ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # fletcher_freeman
        x, f1, g1, niter, neval, duration, x_list, f_list = fletcher_freeman\
            (func,initpoint,eps=1e-15, delta = 1e-8,search_method='armijogoldstain',maxiter=200)
        jsonresult[problem][m]['ff']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,m,' ff ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # Hedben
        x, f1, g1, niter, neval,duration, x_list, f_list = Trust_Region\
            (func,initpoint,eps=1e-15, deltak=1e-1, epsilon=1e-8, maxiter=200,method = 'Hebden')
        jsonresult[problem][m]['H']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,m,' H ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # Cauchy
        x, f1, g1, niter, neval,duration, x_list, f_list = Trust_Region\
            (func,initpoint,eps=1e-15, deltak=1e-1, epsilon=1e-8, maxiter=1000,method = 'Cauchy')
        jsonresult[problem][m]['C']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,m,' C ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # DogLeg
        x, f1, g1, niter, neval,duration, x_list, f_list = Trust_Region\
            (func,initpoint,eps=1e-15, deltak=1e-1, epsilon=1e-8, maxiter=200,method = 'DogLeg')
        jsonresult[problem][m]['DL']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,m,' DL ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # 2DSubMin
        x, f1, g1, niter, neval,duration, x_list, f_list = Trust_Region\
            (func,initpoint,eps=1e-15, deltak=1e-1, epsilon=1e-8, maxiter=200,method = '2DSubMin')
        jsonresult[problem][m]['2DSM']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,m,' 2DSM ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

    # Biggs_EXP6_function
    problem='be6'
    jsonresult[problem]={}

    for m in [8,9,10,11,12]:
        jsonresult[problem][m]={}
        func = Biggs_EXP6_function(6,m)
        initpoint=np.array([1,2,1,1,1,1])

        # damped_Newton
        x, f1, g1, niter, neval, duration, x_list, f_list = damped_Newton\
            (func,initpoint,eps=1e-15, delta = 1e-8,search_method='armijogoldstain',maxiter=200)
        jsonresult[problem][m]['dN']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,m,' dN ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # stable_Newton
        x, f1, g1, niter, neval, duration, x_list, f_list = stable_Newton\
            (func,initpoint,eps=1e-15, delta = 1e-8,search_method='armijogoldstain',maxiter=200)
        jsonresult[problem][m]['sN']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,m,' sN ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # fletcher_freeman
        x, f1, g1, niter, neval, duration, x_list, f_list = fletcher_freeman\
            (func,initpoint,eps=1e-15, delta = 1e-8,search_method='armijogoldstain',maxiter=200)
        jsonresult[problem][m]['ff']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,m,' ff ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # Hedben
        x, f1, g1, niter, neval,duration, x_list, f_list = Trust_Region\
            (func,initpoint,eps=1e-15, deltak=1e-1, epsilon=1e-8, maxiter=200,method = 'Hebden')
        jsonresult[problem][m]['H']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,m,' H ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # Cauchy
        x, f1, g1, niter, neval,duration, x_list, f_list = Trust_Region\
            (func,initpoint,eps=1e-15, deltak=1e-1, epsilon=1e-8, maxiter=200,method = 'Cauchy')
        jsonresult[problem][m]['C']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,m,' C ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # DogLeg
        x, f1, g1, niter, neval,duration, x_list, f_list = Trust_Region\
            (func,initpoint,eps=1e-15, deltak=1e-1, epsilon=1e-8, maxiter=200,method = 'DogLeg')
        jsonresult[problem][m]['DL']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,m,' DL ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

        # 2DSubMin
        x, f1, g1, niter, neval,duration, x_list, f_list = Trust_Region\
            (func,initpoint,eps=1e-15, deltak=1e-1, epsilon=1e-8, maxiter=200,method = '2DSubMin')
        jsonresult[problem][m]['2DSM']={'x':x.tolist(),'f1':f1,'g1':g1.tolist(),'niter':niter,\
                'neval':neval,'duration':duration,'f_list':f_list}
        print(problem,m,' 2DSM ','fvalue:',f1,'niter:',niter,'duration:', duration,sep=' ')

    with open('result.json','w') as f:
        json.dump(jsonresult,f)