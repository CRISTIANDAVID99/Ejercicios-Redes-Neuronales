import pandas as pd
import numpy as np
import scipy.io 

mat=scipy.io.loadmat('set1.mat')
bdr=pd.Series(mat)
bd=pd.DataFrame({'label':bdr.index,'list':bdr.values})
print(bd.iloc[3,:])