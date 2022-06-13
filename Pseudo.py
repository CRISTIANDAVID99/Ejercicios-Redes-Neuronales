import pandas as pd
import numpy as np
import scipy.io 

mat=scipy.io.loadmat('set1.mat')
bdr=pd.Series(mat)
bd=pd.DataFrame({'label':bdr.index,'list':bdr.values})
Xa=pd.DataFrame(bd.iloc[3,1])
Xb=pd.DataFrame(bd.iloc[4,1])
Xa[2]=1
Xb[2]=-1
X=pd.concat([Xa,Xb],axis=0)
print(X)