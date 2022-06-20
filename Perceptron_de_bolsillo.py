from email import header
import pandas as pd
import numpy as np
import scipy.io 
from matplotlib import pyplot as plt 

mat=scipy.io.loadmat('set1.mat')
bdr=pd.Series(mat)
bd=pd.DataFrame({'label':bdr.index,'list':bdr.values})

x0=pd.DataFrame(np.ones(1000))
Xa=pd.DataFrame(bd.iloc[3,1])
Xb=pd.DataFrame(bd.iloc[4,1])

X=pd.concat([Xa,Xb],axis=0).reset_index(drop=True)
X=pd.concat([x0,X],axis=1)
X.columns=range(X.shape[1])

Ya=pd.DataFrame(np.ones(500))
Yb=pd.DataFrame(-np.ones(500))

Y=pd.concat([Ya,Yb],axis=0).reset_index(drop=True)

W=pd.DataFrame(np.random.rand(1,3))
M=0
Mm=0
Wm=W
for i in range(1000):

    ind=np.random.randint(999)
    Yi=pd.concat([Y.iloc[ind]]*3).reset_index(drop=True)

    g=W.dot(X.iloc[ind,:])
    yt=g/abs(g)

    if all(yt*Y.iloc[ind]>=0):

        M+=1
        
    else:
        if M>Mm:
            Wm=W
            Mm=M
        
        W=W+(Yi*X.iloc[ind,:])
        M=0

print(Wm)
print(Mm)