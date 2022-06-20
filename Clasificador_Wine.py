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
print(bd)

X=bd.iloc[:,1:13]
Y=bd[0]
