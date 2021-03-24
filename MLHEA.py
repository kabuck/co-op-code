# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 11:00:45 2021

@author: Matt
"""

import pandas as pd
df = pd.read_csv("C:/Users/Matt/Documents/1 Kayla's Stuff/Week 4 Materials/ds1.csv")
print(df.shape)
headers = ["Alloy No.", "Al%", "Co%", "Cr%", "Cu%", "Fe%", "Ni%", "Predicted HV"]
df.columns = headers #Changes the headers
df.drop([0,0], inplace=True) #removes the extra headers
print(df.head(5))




#K-Nearest Neighbors
    #Fix the continuous problem**
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

X = df[["Al%", "Co%", "Cr%", "Cu%", "Fe%", "Ni%"]].values.astype(float)
X[0:5]
y = df[["Predicted HV"]].values.astype(float)
y[0:5]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state = 4)
print("Train Set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
k = 6
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
neigh
yhat = neigh.predict(X_test)
yhat[0:5].astype(float)

from sklearn import metrics
print("Train set accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set accuracy: ", metrics.accuracy_score(y_test, yhat))


#Decision Tree 
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

X[0:5]
y = df["Predicted HV"]
y[0:5]

from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X,y,test_size = 0.5, random_state = 6)
Tree = DecisionTreeClassifier(criterion = "entropy", max_depth = 6)
Tree.fit(X_trainset, y_trainset)
predTree = Tree.predict(X_testset)
print(predTree[0:5].astype(float))
print(y_testset[0:5].astype(float))

from sklearn import metrics
import matplotlib.pyplot as plt
print("Decision Tree Accuracy: ", metrics.accuracy_score(y_testset, predTree))





