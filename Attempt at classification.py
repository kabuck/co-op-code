# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 10:30:26 2021

@author: Matt
"""

import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
from sklearn import linear_model

#Import/Download Data

df = pd.read_csv("C:/Users/Matt/Documents/1 Kayla's Stuff/Week 4 Materials/ds1.csv")
print(df.shape)


#Format the data

headers = ["Alloy No.", "Al", "Co", "Cr", "Cu", "Fe", "Ni", "HV"]
df.columns = headers #Changes the headers
df.drop([0,0], inplace=True) #removes the extra headers
print(df.head(5))
df = df[["Al", "Co", 'Cr', 'Cu', "Fe", "Ni", "HV"]].astype("float")
df.info()

'''
ID = ["Class"]
if df[["HV"]].value > 875:
    ID = "High HV"
elif HV.value < (875 - (875*0.10)):
    ID = "Low HV"
else:
    ID = "Average"

ID

'''



import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
from sklearn import linear_model

#Import/Download Data

df = pd.read_csv("C:/Users/Matt/Documents/1 Kayla's Stuff/Week 4 Materials/ds1.csv")
print(df.shape)


#Format the data

headers = ["Alloy No.", "Al", "Co", "Cr", "Cu", "Fe", "Ni", "HV"]
df.columns = headers #Changes the headers
df.drop([0,0], inplace=True) #removes the extra headers
print(df.head(5))
df = df[["Al", "Co", 'Cr', 'Cu', "Fe", "Ni", "HV"]].astype("float")
df.info()

c = df["HV"].values
print(c)

if c.any() >= 875.0:
    c = "High HV"
elif c.any() <= (875 - 87.5):
    c = "Low HV"
else:
    c = "Average HV"

print(c)





import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
from sklearn import linear_model

#Import/Download Data

df = pd.read_csv("C:/Users/Matt/Documents/1 Kayla's Stuff/Week 4 Materials/ds1.csv")
print(df.shape)


#Format the data

headers = ["Alloy No.", "Al", "Co", "Cr", "Cu", "Fe", "Ni", "HV"]
df.columns = headers #Changes the headers
df.drop([0,0], inplace=True) #removes the extra headers
print(df.head(5))
df = df[["Al", "Co", 'Cr', 'Cu', "Fe", "Ni", "HV"]].astype("float")
df.info()

df.loc[df["HV"] <= (775.0 - 77.5), "ID"] = -1
df.loc[df["HV"] < 775.0, "ID"] = 0
df.loc[df["HV"] >= 775.0, 'ID'] = 1
df.loc[df["HV"] >= (775.0 + 77.5), 'ID'] = 2


df


#KNN

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

#Import/Download Data

df = pd.read_csv("C:/Users/Matt/Documents/1 Kayla's Stuff/Week 4 Materials/ds1.csv")
print(df.shape)


#Format the data

headers = ["Alloy No.", "Al", "Co", "Cr", "Cu", "Fe", "Ni", "HV"]
df.columns = headers #Changes the headers
df.drop([0,0], inplace=True) #removes the extra headers
print(df.head(5))
df = df[["Al", "Co", 'Cr', 'Cu', "Fe", "Ni", "HV"]].astype("float")
df.info()

#Add classification column

df.loc[df["HV"] <= (774), "ID"] = "Very Low HV"
df.loc[df["HV"] < 775.0, "ID"] = "Low HV"
df.loc[df["HV"] >= 775.0, 'ID'] = "Average HV"
df.loc[df["HV"] >= (775.0 + 77.5), 'ID'] = "High HV"

df["ID"].value_counts()

X = df[["Al", "Co", "Cr", "Cu", "Fe", "Ni"]].values.astype(float)
df_norm = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

y = df["HV"].values
y

df_norm_train, df_norm_test, y_train, y_test = train_test_split(df_norm, y, test_size=0.2, random_state=4)
print("Train set: ", df_norm_train.shape, y_train.shape)
print("Test set: ", df_norm_test.shape, y_test.shape)

k = 3
neigh = KNeighborsRegressor(n_neighbors = k).fit(df_norm_train, y_train)
neigh

yhat = neigh.predict(df_norm_test)
yhat


from sklearn.metrics import r2_score

print("Residual MSE: %.2f"%np.mean((yhat-y)**2))
print("MAE: %.2f"%np.mean(np.absolute(yhat-y)))
print("R2-Score: %.2f"%r2_score(y, yhat))


#stack overflow + error code into google
