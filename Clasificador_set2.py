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
plt.figure(1)
plt.scatter(Xa[0],Xa[1])
plt.scatter(Xb[0],Xb[1])
plt.scatter(Xc[0],Xc[1])

model = Sequential()
model.add(Dense(3, input_dim=2,activation='sigmoid'))
model.add(Dense(1))
model.compile(loss='mean_squared_error',
              optimizer='SGD',
              metrics=['mse'])

model.fit(Xt, Yt, epochs=5000)

scores = model.evaluate(Xt, Yt)
print(scores)
print (model.predict(Xt).round())
Yf=pd.DataFrame(model.predict(Xt).round())
V=Yf==Yt
print(V)
freq = V.value_counts()
print(freq)

plt.figure(2)
plt.scatter(X.iloc[Yf[Yf[0]==2].index,0],X.iloc[Yf[Yf[0]==2].index,1])
plt.scatter(X.iloc[Yf[Yf[0]==1].index,0],X.iloc[Yf[Yf[0]==1].index,1])
plt.scatter(X.iloc[Yf[Yf[0]==0].index,0],X.iloc[Yf[Yf[0]==0].index,1])
plt.show()
