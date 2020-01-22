import numpy as np
import scipy 
from numpy.linalg import norm
import sympy

def BPdecomposition(A):
    '''Bunch Parlett decomposition
    P^T*A*P = L*D*L^T
    
    Parameters
    ----------
    A: ndarray
        matrix to do Bunch Parlett decomposition 
        
    Returns
    -------
    L,D,P,y
    L: ndarray
        L matrix of Bunch Parlett decomposition 
    D: ndarray
        D matrix of Bunch Parlett decomposition  
    P: ndarray
        P matrix of Bunch Parlett decomposition 
    y: ndarray
        permutation order   
    A_new: ndarray
       new matrix of Bunch Parlett decomposition 
    L_list: list
       L list result
    D_list: list
       D list result
    '''
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]
    A_kminus1 = A.copy()
    y=np.arange(n)
    k=1
    m=0
    L_list = []
    D_list = []
    while True:
        argmax_t = np.argmax([abs(A_kminus1[i][i]) for i in range(n-m)])
        att_abs = abs(A_kminus1[argmax_t][argmax_t])
        if m == n-1:
            als_abs = 0
        else:
            als_abs = max([abs(A_kminus1[i][j]) for i in range(n-m) for j in range(n-m) if i>j])
            (argmax_l,argmax_s) = [(i,j) for i in range(n-m) for j in range(n-m) if( i>j and abs(A_kminus1[i][j])==als_abs)][0]
        if att_abs == 0 and als_abs == 0:
            # construct D,L
            D=np.zeros([n,n])
            L=np.zeros([n,n])
            id=0
            for item in D_list:
                if(item.shape[1]==1):
                    D[id:id+1,id:id+1] = item
                    id+=1
                else:
                    D[id:id+2,id:id+2] = item
                    id+=2
            id=0
            for item in L_list:
                if(item.shape[1]==1):
                    L[id::,id:id+1] = item
                    id+=1
                else:
                    L[id::,id:id+2] = item
                    id+=2 
            P = np.zeros([n,n])        
            for id,item in enumerate(y):
                P[id][item]=1
            P = np.linalg.inv(P)
            return L,D,P,y,L.dot(D).dot(L.T),L_list,D_list
        if att_abs >= 2/3*als_abs    :
            # step 5
            A_kminus1[[0,argmax_t]] = A_kminus1[[argmax_t,0]]
            A_kminus1[:,[0,argmax_t]] = A_kminus1[:,[argmax_t,0]]
            for i in range(k-1):
                offset = m - n + L_list[i].shape[0]
                L_list[i][[offset+0,offset+argmax_t]] = L_list[i][[offset+argmax_t,offset+0]]
            y[[m+0,m+argmax_t]] = y[[m+argmax_t,m+0]]
            Dk = A_kminus1[0:1,0:1]
            Lk = np.concatenate([np.identity(1),A_kminus1[1::,0:1]/Dk[0,0]])
            A_kminus1 = A_kminus1 - np.dot(np.dot(Lk,Dk),Lk.T)
            A_kminus1 = A_kminus1[1::,1::]
            D_list.append(Dk)
            L_list.append(Lk)
            m+=1
            if m == n:
                # construct D,L
                D=np.zeros([n,n])
                L=np.zeros([n,n])
                id=0
                for item in D_list:
                    if(item.shape[1]==1):
                        D[id:id+1,id:id+1] = item
                        id+=1
                    else:
                        D[id:id+2,id:id+2] = item
                        id+=2
                id=0
                for item in L_list:
                    if(item.shape[1]==1):
                        L[id::,id:id+1] = item
                        id+=1
                    else:
                        L[id::,id:id+2] = item
                        id+=2        
                P = np.zeros([n,n])        
                for id,item in enumerate(y):
                    P[id][item]=1
                P = np.linalg.inv(P)
                return L,D,P,y,L.dot(D).dot(L.T),L_list,D_list
            else:
                k+=1
                continue
        else:
            #step 6
            A_kminus1[[0,argmax_s]] = A_kminus1[[argmax_s,0]]
            A_kminus1[:,[0,argmax_s]] = A_kminus1[:,[argmax_s,0]] 
            A_kminus1[[1,argmax_l]] = A_kminus1[[argmax_l,1]]
            A_kminus1[:,[1,argmax_l]] = A_kminus1[:,[argmax_l,1]]             
            for i in range(k-1):
                offset = m - n + L_list[i].shape[0]
                L_list[i][[offset+0,offset+argmax_s]] = L_list[i][[offset+argmax_s,offset+0]]
                L_list[i][[offset+1,offset+argmax_l]] = L_list[i][[offset+argmax_l,offset+1]]
            y[[m+0,m+argmax_s]] = y[[m+argmax_s,m+0]]
            y[[m+1,m+argmax_l]] = y[[m+argmax_l,m+1]]
            Dk = A_kminus1[0:2,0:2]    
            Lk = np.concatenate([np.identity(2),np.dot(A_kminus1[2::,0:2],np.linalg.inv(Dk))])
            A_kminus1 = A_kminus1 - np.dot(np.dot(Lk,Dk),Lk.T)
            A_kminus1 = A_kminus1[2::,2::]
            D_list.append(Dk)
            L_list.append(Lk)
            m+=2
            if m == n:
                # construct D,L
                D=np.zeros([n,n])
                L=np.zeros([n,n])
                id=0
                for item in D_list:
                    if(item.shape[1]==1):
                        D[id:id+1,id:id+1] = item
                        id+=1
                    else:
                        D[id:id+2,id:id+2] = item
                        id+=2
                id=0
                for item in L_list:
                    if(item.shape[1]==1):
                        L[id::,id:id+1] = item
                        id+=1
                    else:
                        L[id::,id:id+2] = item
                        id+=2        
                P = np.zeros([n,n])        
                for id,item in enumerate(y):
                    P[id][item]=1
                P = np.linalg.inv(P)
                return L,D,P,y,L.dot(D).dot(L.T),L_list,D_list
            else:
                k+=1
                continue                
                