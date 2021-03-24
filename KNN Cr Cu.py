'''KNN for Cr/Cu data'''

#Import Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn import preprocessing
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.impute import SimpleImputer

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
Hard = 2
Soft = 1
Average = 0
'''

df.loc[(df["Hardness"] >= 775), "ID"] = 2
df.loc[(df["Hardness"] <= 700), "ID"] = 1
df.loc[((df["Hardness"] > 700) & (df["Hardness"] < 775)), "ID"] = 0

df["ID"].value_counts()

#Check data
print("Dataset Length: ", len(df))
print("Dataset: ", str(df))
print("Dataset Shape: ", df.shape)

#Set X and Y
X = np.asarray(df[["Cr", "Cu"]])
Y = np.asarray(df["ID"])

#Data Imputation
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(df.values)
imputed_data = imputer.transform(df.values)
print(imputed_data)


#Train, test data split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
Y_train = Y_train.ravel()
Y_test = Y_test.ravel()

#KNN Implementation
clf = KNeighborsClassifier(n_neighbors = 5, weights = "uniform", algorithm = "auto")
clf.fit(X_train, Y_train)
test_predict = clf.predict(X_test)
f1score = f1_score(test_predict, Y_test)
print("Test F1 Score: ", f1score)

print(X_test.shape)
print(Y_test.shape)


#Check for the most accurate value of K
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

#Plot the elbow curve
test = Elbow(K)
plt.plot(K, test)
plt.xlabel("K Neighbors")
plt.ylabel("Test MSE")
plt.title("Elbow Curve for Test")
plt.show()

'''
Choose K = 5; MSE = 0; Accuracy = 100%
'''

#Plotting KNN

#take the first two features
X = np.asarray(df[["Cr", "Cu"]])
y = np.asarray(df["ID"])

print(X)
print(y)

h = .02  #step size in the mesh

#Calculate min, max and limits
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#Put the result into a color plot
tnrfont = {'fontname':'Times New Roman', 'color':'black'}

plt.figure()
sns.set_style(style = "white")

plt.scatter(X[:, 0], X[:, 1])
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Data points", **tnrfont)
plt.xlabel("at% Cr", **tnrfont)
plt.ylabel("at% Cu", **tnrfont)
plt.show()

clf = neighbors.KNeighborsClassifier(n_neighbors = 5, weights='distance')
clf.fit(X, y)

print(clf.predict(X))

#Neighbors
n_neighbors = 5

#Create color maps
cmap_light = ListedColormap(['pink', 'lightblue','lightgreen'])
cmap_bold = ListedColormap(['red', 'darkblue','darkgreen'])

#we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, weights='distance')
clf.fit(X, y)

#calculate min, max and limits
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

#predict class using data and kNN classifier
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

#Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

#Plot also the training points
tnrfont = {'fontname':'Times New Roman', 'color':'black'}

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class Classification (k = %i)" % (n_neighbors), **tnrfont)
plt.xlabel('at% Cr', **tnrfont)
plt.ylabel('at% Cu', **tnrfont)
plt.show()

'''Test data points are included in the scatter plot'''

#Predict random composition
Cr = 42
Cu = 5
dataClass = clf.predict([[Cr,Cu]])
print('Prediction: ')

if dataClass == 0:
    print('Average, pink')
elif dataClass == 1:
    print('Soft, blue')
else:
    print('Hard, green')
