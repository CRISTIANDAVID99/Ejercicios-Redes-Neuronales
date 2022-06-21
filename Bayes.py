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
Xat=Xa.iloc[:int(len(Xa.index)*0.8),:]
Xbt=Xb.iloc[:int(len(Xb.index)*0.8),:]

Xav=Xa.iloc[int(len(Xa.index)*0.8):,:]
Xbv=Xb.iloc[int(len(Xb.index)*0.8):,:]

fig, (axs1, axs2)=plt.subplots(1,2)
axs1.scatter(Xav[1],Xav[0],c='b')
axs1.scatter(Xbv[1],Xbv[0],c='r')
axs1.set_title('Datos reales')
Ya=pd.DataFrame(np.ones(100))
Yb=pd.DataFrame(-np.ones(100))
Y=pd.concat([Ya,Yb],axis=0).reset_index(drop=True)

Xv=pd.concat([Xav,Xbv],axis=0).reset_index(drop=True)



Py_c1=len(Xa.index)/len(X.index)


Za=Xat.cov()
Za_m=Za.to_numpy()
Za_inv=np.linalg.inv(Za_m)

Zb=Xbt.cov()
Zb_m=Zb.to_numpy()
Zb_inv=np.linalg.inv(Zb_m)

mua=Xat.mean()
mub=Xbt.mean()

ln_Z=np.log(abs(np.linalg.norm(Za_m))/abs(np.linalg.norm(Za_m)))
xt=Xv.transpose().reset_index(drop=True)
xt.columns=range(xt.columns.size)

rep=len(xt.columns)

Ck=-(1/2)*np.linalg.norm(Za_inv)*(xt-pd.concat([mua]*rep,axis=True)).transpose().dot(xt-pd.concat([mua]*rep,axis=True))+(1/2)*np.linalg.norm(Zb_inv)*(xt-pd.concat([mub]*rep,axis=True)).transpose().dot(xt-pd.concat([mub]*rep,axis=True))+(1/2)*ln_Z-np.log((1-Py_c1)/Py_c1)

A=-(1/2)*np.linalg.norm(Za_inv)
B=(1/2)*np.linalg.norm(Zb_inv)
F=(1/2)*ln_Z
O=(1/2)*ln_Z-np.log((1-Py_c1)/Py_c1)

print(A,'{[x1+(',mua.to_string(),')]^2+[x2+(',mua.to_string(),')]^2}+',B,'{[x1+(',mub.to_string(),')]^2+[x2+(',mub.to_string(),')]^2}+(',O,')')
K=pd.DataFrame(np.diag(Ck))
err = ((K/abs(K))*(Y))
Kerr=err.value_counts()
print("Error de generalizacion: ")
print(Kerr.iloc[-1]/200)

axs2.scatter(Xv.iloc[K[K[0]<0].index,0],Xv.iloc[K[K[0]<0].index,1])
axs2.scatter(Xv.iloc[K[K[0]>=0].index,0],Xv.iloc[K[K[0]>=0].index,1])

axs2.set_title('Datos estimados')
plt.show()