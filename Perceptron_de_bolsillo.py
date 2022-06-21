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

plt.scatter(Xa[1],Xa[0],c='b')
plt.scatter(Xb[1],Xb[0],c='r')

X=pd.concat([Xa,Xb],axis=0).reset_index(drop=True)
X=pd.concat([x0,X],axis=1)
X.columns=range(X.shape[1])

ik=np.random.permutation(X.index)
X=X.iloc[ik,:].reset_index(drop=True) 

Ya=pd.DataFrame(np.ones(500))
Yb=pd.DataFrame(-np.ones(500))

Y=pd.concat([Ya,Yb],axis=0).reset_index(drop=True)
Y=Y.iloc[ik,:].reset_index(drop=True) 

Xt=X.iloc[:int(len(X.index)*0.8)-1,:]
Xv=X.iloc[int(len(X.index)*0.8)-1:,:].reset_index(drop=True)

Yt=Y.iloc[:int(len(Y.index)*0.8)-1,:]
Yv=Y.iloc[int(len(Y.index)*0.8)-1:,:].reset_index(drop=True)

W=pd.DataFrame(np.random.rand(1,3))
M=0
Mm=0
Wm=W
for i in range(1000):

    ind=np.random.randint(799)
    Yi=pd.concat([Yt.iloc[ind]]*3).reset_index(drop=True)

    g=W.dot(Xt.iloc[ind,:])
    yt=g/abs(g)

    if all(yt*Yt.iloc[ind]>=0):

        M+=1
        
    else:
        if M>Mm:
            Wm=W
            Mm=M
        
        W=W+(Yi*Xt.iloc[ind,:])
        M=0

print(Wm)
print(Mm)

ft=W.iloc[0,0]+W.iloc[0,1]*Xv.iloc[:,1]+W.iloc[0,2]*Xv.iloc[:,0]
ft=ft.reset_index(drop=True)
ft=ft.to_frame()
err = ((ft/abs(ft))*(Yv))
Kerr=err.value_counts()
print("Error de generalizacion: ")
print(Kerr.iloc[-1]/200)


xs=np.linspace(-5,5)
fy=-(W.iloc[0,0]+W.iloc[0,1]*xs)/W.iloc[0,2]
plt.plot(xs,fy)

plt.title("SET1")
plt.legend(['C1','C2'])
plt.show()