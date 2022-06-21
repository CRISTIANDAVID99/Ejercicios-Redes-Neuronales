import pandas as pd
import numpy as np
import scipy.io 
from matplotlib import pyplot as plt

mat=scipy.io.loadmat('set1.mat')
bdr=pd.Series(mat)
bd=pd.DataFrame({'label':bdr.index,'list':bdr.values})
Xa=pd.DataFrame(bd.iloc[3,1])
Xb=pd.DataFrame(bd.iloc[4,1])
Xa[3]=1
Xb[3]=-1

X=pd.concat([Xa,Xb],axis=0).reset_index(drop=True)

X[2]=1 #X=[X2,X1,X0,Y]

ik=np.random.permutation(X.index)
X=X.iloc[ik,:].reset_index(drop=True) 

Xt=X.iloc[:int(len(X.index)*0.8),:]
Xv=X.iloc[int(len(X.index)*0.8):,:]

plt.scatter(Xa[1],Xa[0],c='b')
plt.scatter(Xb[1],Xb[0],c='r')


W=Xt.iloc[:,[0,1,2]].transpose().dot(Xt.iloc[:,[0,1,2]]).dot(Xt.iloc[:,[0,1,2]].transpose()).dot(Xt[3]) #Pseundo-inversa
print("Pesos: ")
print(W)
xs=np.linspace(-5,5)
fy=-(W.iloc[2]+W.iloc[1]*xs)/W.iloc[0]
plt.plot(xs,fy)
ft=W.iloc[2]+W.iloc[1]*Xv.iloc[:,1]+W.iloc[0]*Xv.iloc[:,0]
ft=ft.reset_index(drop=True)


err = ((ft/abs(ft))*(Xv.iloc[:,3].reset_index(drop=True)))
Kerr=err.value_counts()
print("Error de generalizacion: ")
print(Kerr.iloc[-1]/200)
plt.title("SET1")
plt.legend(['C1','C2'])
plt.show()