from email import header
import pandas as pd
import numpy as np
import scipy.io 
from matplotlib import pyplot as plt

mat=scipy.io.loadmat('set1.mat')
bdr=pd.Series(mat)
bd=pd.DataFrame({'label':bdr.index,'list':bdr.values})
Xa=pd.DataFrame(bd.iloc[3,1])
Xb=pd.DataFrame(bd.iloc[4,1])

X=pd.concat([Xa,Xb],axis=0)

plt.scatter(Xa[1],Xa[0],c='b')
plt.scatter(Xb[1],Xb[0],c='r')

Py_c1=len(Xa.index)/len(X.index)


Za=Xa.cov()
Za_m=Za.to_numpy()
Za_inv=np.linalg.inv(Za_m)
Zb=Xb.cov()
Zb_m=Zb.to_numpy()
Zb_inv=np.linalg.inv(Zb_m)
mua=Xa.mean()
mub=Xb.mean()
ln_Z=np.log(abs(np.linalg.norm(Za_m))/abs(np.linalg.norm(Za_m)))
xt=Xb.iloc[10,:].transpose()

Ck=-(1/2)*np.linalg.norm(Za_inv)*(xt-mua).transpose().dot(xt-mua)+(1/2)*np.linalg.norm(Zb_inv)*(xt-mub).transpose().dot(xt-mub)+(1/2)*ln_Z
print(Ck)
K=-1 if Ck<=np.log((1-Py_c1)/Py_c1) else 1
print(K)