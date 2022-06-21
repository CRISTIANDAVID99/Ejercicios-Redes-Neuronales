from ast import Import
from email import header
from itertools import accumulate
from re import A
import pandas as pd
import numpy as np
import scipy.io 
from matplotlib import pyplot as plt
import keras.optimizer_v1
from keras.models import Sequential
from keras.layers.core import Dense
import keras.optimizer_v1

mat=scipy.io.loadmat('set2.mat')
bdr=pd.Series(mat)
bd=pd.DataFrame({'label':bdr.index,'list':bdr.values})

keras.optimizer_v1.SGD(lr=0.1)

Xa=pd.DataFrame(bd.iloc[3,1])
Xb=pd.DataFrame(bd.iloc[4,1])
Xc=pd.DataFrame(bd.iloc[5,1])

X=pd.concat([Xa,Xb,Xc],axis=0).reset_index(drop=True)


Ya=pd.DataFrame(2*np.ones(200))
Yb=pd.DataFrame(np.ones(200))
Yc=pd.DataFrame(np.zeros(200))

Y=pd.concat([Ya,Yb,Yc],axis=0).reset_index(drop=True)

ind=np.random.permutation(X.index)
X=X.iloc[ind].reset_index(drop=True)
Y=Y.iloc[ind].reset_index(drop=True)

Xt=X.iloc[X.index<int(len(X.index)*0.8)]
Yt=Y.iloc[Y.index<int(len(Y.index)*0.8)]

Xv=X.iloc[int(len(X.index)*0.8):,:].reset_index(drop=True)
Yv=Y.iloc[int(len(Y.index)*0.8):,:].reset_index(drop=True)

k=0
plt.figure(1)
plt.scatter(Xa[0],Xa[1])
plt.scatter(Xb[0],Xb[1])
plt.scatter(Xc[0],Xc[1])
plt.title("SET 2")
for i in np.arange(0.001,2,0.25):
    keras.optimizer_v1.SGD(lr=(i))
    model = Sequential()
    model.add(Dense(3, input_dim=2,activation='sigmoid'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',
                optimizer='SGD',
                metrics=['mse'])

    hisy=model.fit(Xt, Yt, epochs=1500)

    scores = model.evaluate(Xt, Yt)
    plt.figure(3)
    plt.plot(hisy.history['loss'])
    plt.xlabel('epoch')
    plt.xlabel('ECM')
    plt.title('ECM vs. epochs')
    fli="{:.3f}".format(i)
    k=k+1

plt.legend(['0.001','0.251','0.501','0.751','1.001','1.251','1.501','1.751'])

print(scores)
print (model.predict(Xv).round())
Yf=pd.DataFrame(model.predict(Xv).round())
print(Yv)
V=Yf==Yv
print(V)
freq = V.value_counts()
print(freq)

plt.figure(2)
plt.scatter(Xv.iloc[Yf[Yf[0]==2].index,0],Xv.iloc[Yf[Yf[0]==2].index,1])
plt.scatter(Xv.iloc[Yf[Yf[0]==1].index,0],Xv.iloc[Yf[Yf[0]==1].index,1])
plt.scatter(Xv.iloc[Yf[Yf[0]==0].index,0],Xv.iloc[Yf[Yf[0]==0].index,1])
plt.title("Clasificacion SET 2")



plt.show()
