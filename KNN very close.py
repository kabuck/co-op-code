# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:40:31 2021

@author: Matt
"""
#Import Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
'''
from sklearn.preprocessing import Imputer
did not work, try it a different way
'''
from sklearn.impute import SimpleImputer
'''
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(df.values)
imputed_data = imputer.transform(df.values)
print(imputed_data)
'''

#Import/Download Data

df = pd.read_csv("C:/Users/Matt/Documents/1 Kayla's Stuff/Week 4 Materials/ds1.csv")
print(df.shape)
df.head(5)

#Format the data

headers = ["Alloy No.", "Al", "Co", "Cr", "Cu", "Fe", "Ni", "Hardness"]
df.columns = headers #Changes the headers
df.drop([0,0], inplace=True) #removes the extra headers
print(df.head(5))
df = df[["Al", "Co", 'Cr', 'Cu', "Fe", "Ni", "Hardness"]].astype("float")
df.info()
print(df.min)
df.head(5)



#Add categorical data: hard or soft
'''
if df["Hardness"] >= 850:
    df.loc[df["Hardness"] >= 850, "Category"] = "Hard"
elif df["Hardness"] <= 700:
    df.loc[df["Hardness"] >= 700, "Category"] = "Soft"
elif df["Hardness"] > 700 and df["Hardness"] < 850:
    df.loc[((df["Hardness"] > 700) & (df["Hardness"] < 850)), "Category"] = "Average"

df["Category"].value_counts()
'''
'''
if df["Hardness"] >= 850:
    headers2 = ["Alloy No.", "Al", "Co", "Cr", "Cu", "Fe", "Ni", "Hardness", "ID"]
    df.columns = headers2
    df.loc[(df["Hardness"] >= 850), "ID"] = "Hard"
    print(df.head(5))
elif df["Hardness"] <= 700:
    headers2 = ["Alloy No.", "Al", "Co", "Cr", "Cu", "Fe", "Ni", "Hardness", "ID"]
    df.columns = headers2
    df.loc[(df["Hardness"] <= 700), "ID"] = "Soft"
    print(df.head(5))
elif (df["Hardness"] > 700) & (df["Hardness"] < 850):
    headers2 = ["Alloy No.", "Al", "Co", "Cr", "Cu", "Fe", "Ni", "Hardness", "ID"]
    df.columns = headers2
    df.loc[((df["Hardness"] > 700) & (df["Hardness"] < 850)), "ID"] = "Average"
    print(df.head(5))
'''
#Hard = 2
#Soft = 1
#Average = 0

df.loc[(df["Hardness"] >= 850), "ID"] = 2
df.loc[(df["Hardness"] <= 700), "ID"] = 1
df.loc[((df["Hardness"] > 700) & (df["Hardness"] < 850)), "ID"] = 0

df["ID"].value_counts()

#Check data
print("Dataset Length: ", len(df))
print("Dataset: ", str(df))
print("Dataset Shape: ", df.shape)

#Set X and Y
X = df.drop(["ID"], axis = 1)
Y = df["ID"]
print(X.head(5), "\n", X.shape)
print(Y.head(5), "\n", Y.shape)
'''
#Data Imputation
imp = SimpleImputer(missing_values = "NaN", strategy = "median", axis = 0)
X = imp.fit_transform(X)
'''
#Train, test data split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
Y_train = Y_train.ravel()
Y_test = Y_test.ravel()

#KNN Implementation
from sklearn.neighbors import KNeighborsRegressor as KNNR
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

def Elbow(K):
    test_mse = []
    
    for i in K:
        K_value = i
        neigh = KNeighborsClassifier(n_neighbors = K_value, weights ="uniform", algorithm = "auto")
        neigh.fit(X_train, Y_train)
        y_pred = neigh.predict(X_test)
        tmp1 = mse(y_pred, Y_test)
        test_mse.append(tmp1)
        print("Accuracy is ", accuracy_score(Y_test, y_pred)*100, "% for K-Value: ", K_value)

    return test_mse

K = range(1,25)

test = Elbow(K)
plt.plot(K, test)
plt.xlabel("K Neighbors")
plt.ylabel("Test MSE")
plt.title("Elbow Curve for Test")
plt.show()

'''
Choose K = 11; MSE = 0; Accuracy = 100%
'''

#Trying to plot KNN
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 11

X = df[:, :2]
Y = df["ID"]

h = 0.02

cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
cmap_bold = ["darkorange", "c", "darkblue"]

for weights in ["uniform", "distance"]:
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights = weights)
    clf.fit(X,Y)
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    Z = Z.reshape(xx.shape)
    plt.figure(figsize = (8,6))
    plt.contourf(xx, yy, Z, cmap = cmap_light)
    
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=df[Y], palette=cmap_bold, alpha = 1.0, edgecolor = "black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class Classification (k = %i, weights = '%s')" % (n_neighbors, weights))
    plt.xlabel(df.feature_names[0])
    plt.ylabel(df.feature_names[1])
    
plt.show()
'''
'''
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, neighbors
from mlxtend.plotting import plot_decision_regions


def knn_comparison(data, k):
    X = df[["Al", "Co", 'Cr', 'Cu', "Fe", "Ni", "Hardness"]].values
    Y = df["ID"].astype(int).values
    clf = neighbors.KNeighborsClassifier(n_neighbors = K)
    clf.fit(X, Y)
    
    plot_decision_regions(X, Y, clf = clf, legend = 2)
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("KNN with K = " + str(K))
    plt.show()
'''
   
    
import matplotlib
#matplotlib.use('GTK3Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

# import some data to play with

# take the first two features
X = np.asarray(df[["Al", "Co"]])
y = np.asarray(df["ID"])

print(X)
print(y)


import matplotlib
#matplotlib.use('GTK3Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


# take the first two features
X = np.array(df[["Al", "Co"]])
y = np.array(df["ID"])
print(y)
h = .02  # step size in the mesh

# Calculate min, max and limits
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Put the result into a color plot
plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Data points")
plt.show()


clf = neighbors.KNeighborsClassifier(n_neighbors = 11, weights='distance')
clf.fit(X, y)

print(clf.predict(X))


import matplotlib
#matplotlib.use('GTKAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 11


# prepare data
X = np.array(df[["Al", "Co"]])
y = np.array(df["ID"])
h = .02

# Create color maps
cmap_light = ListedColormap(['gold', 'orange','yellow'])
cmap_bold = ListedColormap(['cyan', 'blue','purple'])

# we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(X, y)

# calculate min, max and limits
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

# predict class using data and kNN classifier
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i)" % (n_neighbors))
plt.show()

'''
import numpy as np
from sklearn import neighbors, datasets
from sklearn import preprocessing

n_neighbors = 11


# prepare data
X = np.array(df[["Al", "Hardness"]])
y = np.array(df["ID"])
h = .02

# we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, weights='distance')
clf.fit(X, y)
'''

'''
# make prediction
sl = raw_input('Enter sepal length (cm): ')
sw = raw_input('Enter sepal width (cm): ')
dataClass = clf.predict([[sl,sw]])
print('Prediction: '),

if dataClass == 0:
    print('Iris Setosa')
elif dataClass == 1:
    print('Iris Versicolour')
else:
    print('Iris Virginica')
'''

