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
X=pd.concat([Xa,Xb],axis=0)
X[2]=1 #X=[X2,X1,X0,Y]
plt.scatter(Xa[1],Xa[0],c='b')
plt.scatter(Xb[1],Xb[0],c='r')


W=X.iloc[:,[0,1,2]].transpose().dot(X.iloc[:,[0,1,2]]).dot(X.iloc[:,[0,1,2]].transpose()).dot(X[3]) #Pseundo-inversa
print(W)
xs=np.linspace(-5,5)
fy=-(W.iloc[2]+W.iloc[1]*xs)/W.iloc[0]
plt.plot(xs,fy)
plt.show()