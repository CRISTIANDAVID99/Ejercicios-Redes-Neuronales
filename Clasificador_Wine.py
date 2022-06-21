from ast import For
from calendar import c
import csv
from tkinter import Y
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import keras.optimizer_v1
from keras.models import Sequential
from keras.layers.core import Dense

#carga de base de datos 
bd=pd.read_csv("wine.csv",header=None)

X=bd.iloc[:,1:14]
Y=bd.iloc[:,0].reset_index(drop=True)
Y=Y.to_frame()
ind=np.random.permutation(X.index)
X=X.iloc[ind].reset_index(drop=True)
Y=Y.iloc[ind].reset_index(drop=True)

Xt=X.iloc[X.index<int(len(X.index)*0.8)]
Yt=Y.iloc[Y.index<int(len(Y.index)*0.8)]

keras.optimizer_v1.Adam(lr=0.1)

err = pd.DataFrame(columns=['false',],
                  index=range(20))
for i in range(10):
    model = Sequential()
    model.add(Dense(15, input_dim=13,activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',
                optimizer='Adam',
                metrics=['mse'])

    model.fit(Xt, Yt, epochs=2000)

    scores = model.evaluate(Xt, Yt)
    print(scores)
    print (model.predict(Xt).round())
    Yf=pd.DataFrame(model.predict(Xt).round())
    V = (Yt.iloc[:]==Yf.iloc[:])

    freq = V.value_counts()
    print(freq.keys())
    print(err)
    err.iloc[i]=freq[False]/140

print("Error de generaliacion: ")
print(err)
