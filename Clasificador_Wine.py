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
model = Sequential()
model.add(Dense(50, input_dim=13,activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error',
              optimizer='Adam',
              metrics=['mse'])

model.fit(Xt, Yt, epochs=5000)

scores = model.evaluate(Xt, Yt)
print(scores)
print (model.predict(Xt).round())
Yf=pd.DataFrame(model.predict(Xt).round())
V = (Yt.iloc[:]==Yf.iloc[:])

print(V)
freq = V.value_counts()
print(freq)
print(Yf)
print(Yt)
