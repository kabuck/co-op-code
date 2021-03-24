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
import mlxtend


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

df.max


#Add categorical data: hard or soft
'''
Hard = 2 = green
Soft = 1 = blue
Average = 0 = red
'''

df.loc[(df["Hardness"] >= 700), "ID"] = 2
df.loc[(df["Hardness"] <= 550), "ID"] = 1
df.loc[((df["Hardness"] > 550) & (df["Hardness"] < 700)), "ID"] = 0

df["ID"].value_counts()

df = df[["Al", "Co", 'Cr', 'Cu', "Fe", "Ni", "Hardness", "ID"]].astype("float")
print(df.head(5))


#Defining dataset

#First Feature
al = np.array(df["Al"])

#Second Feature
cr = np.array(df["Cr"])

#Third Feature
co = np.array(df["Co"])

#Fourth Feature
cu = np.array(df["Cu"])

#Fifth Feature
fe = np.array(df["Fe"])

#Sixth Feature
ni = np.array(df["Ni"])


#Lable or target variable
hardness_class = np.array(df["ID"])


#Combining Features

features = np.array(list(zip(al, cr, co, cu, fe, ni)))

print(features)


''' At this point, X = features, Y = hardness_class '''


#Splitting Data
'''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, hardness_class, test_size=0.3)
'''

import numpy as np
from sklearn.model_selection import KFold
X = features
y = hardness_class
n_splits = 10
kf = KFold(n_splits=n_splits)
kf.get_n_splits(X)

print(kf)
KFold(n_splits=n_splits, random_state=None, shuffle=False)
for train_index, test_index in kf.split(X):
     print("TRAIN:", train_index, "TEST:", test_index)
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]


#Generating MOdel for K = 5

from sklearn.neighbors import KNeighborsClassifier as KNC

knn = KNC(n_neighbors = 4)

#train the model using the trianing sets
knn.fit(X_train, y_train)

#predict the response for test dataset
y_pred = knn.predict(X_test)

X_test.shape

#Model Evaluation for k = 5
from sklearn import metrics

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))


'''
X = features
Y = hardness_class


#Train, test data split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
Y_train = Y_train.ravel()
Y_test = Y_test.ravel()

#KNN Implementation
clf = KNeighborsClassifier(n_neighbors = 3, weights = "uniform", algorithm = "auto")
clf.fit(X_train, Y_train)
test_predict = clf.predict(X_test)
f1score = f1_score(test_predict, Y_test)
print("Test F1 Score: ", f1score)

print(X_test.shape)
print(Y_test.shape)
'''

#Check for the most accurate value of K
def Elbow(K):
    test_mse = []
    
    for i in K:
        K_value = i
        neigh = KNeighborsClassifier(n_neighbors = K_value, weights ="uniform", algorithm = "auto")
        neigh.fit(X_train, y_train)
        y_pred = neigh.predict(X_test)
        tmp1 = mse(y_pred, y_test)
        test_mse.append(tmp1)
        print("Accuracy is ", accuracy_score(y_test, y_pred)*100, "% for K-Value: ", K_value)

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


#Splitting Data

import numpy as np
from sklearn.model_selection import KFold
X = features
y = hardness_class
kf = KFold(n_splits=n_splits)
kf.get_n_splits(X)

print(kf)
KFold(n_splits=n_splits, random_state=None, shuffle=False)
for train_index, test_index in kf.split(X):
     print("TRAIN:", train_index, "TEST:", test_index)
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]

#Generating Model for K = 6

from sklearn.neighbors import KNeighborsClassifier as KNC

knn = KNC(n_neighbors = 4)

#train the model using the trianing sets
knn.fit(X_train, y_train)

#predict the response for test dataset
y_pred = knn.predict(X_test)


#Model Evaluation for k = 4
from sklearn import metrics

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

























'''
#Predict random composition

predicted_df = pd.read_csv("C:/Users/Matt/Documents/1 Kayla's Stuff/Week 8 Materials/predicted HV.csv")

predicted_df.head(5)

al_p = np.array(predicted_df["Al"])
cr_p = np.array(predicted_df["Co"])
co_p = np.array(predicted_df["Cr"])
cu_p = np.array(predicted_df["Cu"])
fe_p = np.array(predicted_df["Fe"])
ni_p = np.array(predicted_df["Ni"])
hardness_class_p = np.array(Y_train)

predict_features = np.array(list(zip(al_p, cr_p, co_p, cu_p, fe_p, ni_p))

y_pred2 = knn.predict(predict_features)

print(y_pred2)

dataClass = knn.predict(predict_features)
print('Prediction: ')

if dataClass == 0:
    print('Average, pink')
elif dataClass == 1:
    print('Soft, blue')
else:
    print('Hard, green')
'''




#Correlation Matrix Heatmap (Including Hardness as a Feature)
tnrfont = {'fontname':'Times New Roman', 'color':'black'}


f, ax = plt.subplots(figsize=(10,6))
all_features = df[["Al", "Cr", "Co", "Cu", "Fe", "Ni", "Hardness"]]
corr = all_features.corr()
heatmap = sns.heatmap(round(corr, 2), annot=True, ax=ax, cmap="coolwarm", fmt=".2f", linewidths="0.5")
f.subplots_adjust(top=0.93)
t = f.suptitle("Alloy Attributes Correlation Heatmap (with Hardness)", fontsize=14, **tnrfont)

#Correlation Matrix Heatmap
tnrfont = {'fontname':'Times New Roman', 'color':'black'}


f, ax = plt.subplots(figsize=(10,6))
features_df = df[["Al", "Cr", "Co", "Cu", "Fe", "Ni"]]
corr = features_df.corr()
heatmap = sns.heatmap(round(corr, 2), annot=True, ax=ax, cmap="coolwarm", fmt=".2f", linewidths="0.5")
f.subplots_adjust(top=0.93)
t = f.suptitle("Alloy Attributes Correlation Heatmap", fontsize=14, **tnrfont)


#Pair-Wise Scatter Plots

columns = features
pp = sns.pairplot(features_df, size=1.8, aspect =1.8, plot_kws=dict(edgecolor="k", linewidth=0.5), diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle("Alloy Attributes Pairwise Plots", fontsize=14)



'''
The graphs along the diagonal shows the distribution of a single variable
    It can be useful in determining if the data of a single variable is normally distributed or skewed;
    Cr and maybe Al seems to have a left hand skew, while Ni and Fe seem to have the strongest normal distribution.
    Cu looks more normally distributed

The other graphs show the relationship between different variables;
those in the top right and bottom left are the same, just different orientation.
'''






'''

Can't figure out how to combine the scaled data and class ID
successfully scaled data though



#Scaling Attribute Values to avoid Outliers
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

features_df_wID = df[["Al", "Cr", "Co", "Cu", "Fe", "Ni", "ID"]]
scaled_features_df = ss.fit_transform(features_df_wID)
scaled_features_df = pd.DataFrame(scaled_features_df, columns=["Al", "Cr", "Co", "Cu", "Fe", "Ni"])
final_df = pd.concat([scaled_features_df, df["ID"]], axis=1)

final_df.head(5)

#plot parallel coordinates
from pandas.plotting import parallel_coordinates
parallel_coord = parallel_coordinates(final_df, "ID", color=("red", "blue", "green"))
'''
'''
cols_all = ["Al", "Cr", "Co", "Cu", "Fe", "Ni", "ID"]
total_df = df[cols_all]

total_df.head(5)
cols = ["Al", "Cr", "Co", "Cu", "Fe", "Ni"]
subset_df = total_df[cols]


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

scaled_df = ss.fit_transform(subset_df)
scaled_df = pd.DataFrame(scaled_df, columns=cols)
scaled_df.head(5)

#Add class ID
final_df = pd.concat([scaled_df, df["ID"]])

headers = ["ID", "Al", "Co", "Cr", "Cu", "Fe", "Ni"]
final_df.columns=headers

final_df = pd.concat([scaled_df[cols], df["ID"]])
final_df.head(5)

headers = ["ID", "Al", "Co", "Cr", "Cu", "Fe", "Ni"]
final_df.columns = headers
final_df.head(5)

#Trying to reindex ID
df2 = df[["Hardness", "ID"]]
dfrange = pd.DataFrame(range(0, 155, 1))
dfrange.columns = ["range"]

dfrange.tail(5)

class_array = np.array(df2["ID"])
range_array = np.array(dfrange)
group_array = np.array(list(zip(class_array, range_array)))

group_df = pd.DataFrame(group_array, columns=["ID", "Range"])

group_df.head(5)

final_df = group_df["ID"].replace(final_df["ID"])

final_df.head(5)


'''
'''
#Convert Scaled Data to Arrays
scaled_df_array1 = np.array(scaled_df["Al"])
scaled_df_array2 = np.array(scaled_df["Cr"])
scaled_df_array3 = np.array(scaled_df["Co"])
scaled_df_array4 = np.array(scaled_df["Cu"])
scaled_df_array5 = np.array(scaled_df["Fe"])
scaled_df_array6 = np.array(scaled_df["Ni"])
class_array = np.array(total_df["ID"])

#Combine All Arrays into 1 Array
final_array = np.array(list(zip(scaled_df_array1, scaled_df_array2, scaled_df_array3, scaled_df_array4, scaled_df_array5, scaled_df_array6, class_array)))

#Convert back to a dataframe
final_df = pd.DataFrame(final_array, columns = [cols_all])
final_df.head(5)


#Check for missing Data
missing_data = final_df.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")
'''
'''
#Plot Parallel Coordinates
from pandas.plotting import parallel_coordinates
pc = parallel_coordinates(final_df, "ID", color=[("red", "blue", "green")])
'''



df.head(5)

#Pair-Wise Scatter Plot 3-D

col = ["Al", "Co", "Cr", "Cu", "Fe", "Ni", "ID"]
pp = sns.pairplot(df[col], hue = "ID", height = 2, aspect = 2, palette={0.0: "red", 1.0: "blue", 2.0:"green"}, plot_kws=dict(edgecolor = "black", linewidth=0.5))

fig = pp.fig
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle("Alloy Features Pairwise Plots", fontsize = 30, **tnrfont)










# Visualizing 3-D mix data using scatter plots
# leveraging the concepts of hue for categorical dimension
jp = sns.pairplot(df, x_vars=["Fe"], y_vars=["Cr"], size=4.5,
                  hue="ID", palette={0.0: "red", 1.0: "blue", 2.0: "green"},
                  plot_kws=dict(edgecolor="k", linewidth=0.5))
                  
# we can also view relationships\correlations as needed                  
lp = sns.lmplot(x='Fe', y='Cr', hue='ID', 
                palette={0.0: "red", 1.0: "blue", 2.0: "green"},
                data=df, fit_reg=True, legend=True,
                scatter_kws=dict(edgecolor="k", linewidth=0.5))






from sklearn import linear_model
samples = np.random.rand(len(df))<0.8
train = df[samples] #80% of data used to train the model
test = df[~samples] #remainder of data used to test the model

#Train Data Distribution

plt.scatter(train.Fe, train.Cr, color = "black", alpha = 0.5)
plt.xlabel("at% Fe")
plt.ylabel("at% Cr")
plt.title("at% Fe vs. Cr Training Data")

plt.show() #prints the raw data

regr1 = linear_model.LinearRegression() 
train_x = np.asanyarray(train[["Fe"]]) #Converts the training data (IV) to an array
train_y = np.asanyarray(train[["Cr"]]) #Converts training data (DV) to an array
regr1.fit(train_x, train_y) #fits a line of best fit to the data

print("Coefficients: ", regr1.coef_) #Slope of the regression line
print("Intercept: ", regr1.intercept_) #Intercept of the regression line

plt.scatter(train.Fe, train.Cr, color = "black") #plots trianing data
plt.plot(train_x, regr1.coef_[0][0]*train_x + regr1.intercept_[0]) #plots the regression line
plt.xlabel("at% Fe")
plt.ylabel("at% Cr")
plt.title("Multiple Linear Regression Model: at% Fe vs. at% Cr")

plt.show()

from sklearn.metrics import r2_score 

test_x = np.asanyarray(test[["Fe"]]) #Converts testing data (IV) to an array
test_y = np.asanyarray(test[["Cr"]]) #Converts testing data (DV) to an array
test_1 = regr1.predict(test_x) #Predicts the HV of the testing values using the regression equation

#Accuracy Metrics
print("Mean Absolute Error: %.2f"%np.mean(np.absolute(test_1-test_y)))
print("Residual Mean Squared Error: %.2f"%np.mean((test_1-test_y)**2))
print("R2 Score: %.2f"%r2_score(test_y, test_1))

















# Visualizing 3-D numeric data with Scatter Plots
# length, breadth and depth
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

xs = df['Al']
ys = df['Co']
zs = df['Cr']
ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w')

ax.set_xlabel('at% Al', **tnrfont)
ax.set_ylabel('at% Co', **tnrfont)
ax.set_zlabel('at% Cr', **tnrfont)
ax.set_title("Visualization of 3 Continuous, Numeric Features (Al, Co, Cr)", **tnrfont)

'''
# Visualizing 4-D numeric data with Scatter Plots (ATTEMPT)
# length, breadth and depth

#Create color maps
cmap_light = ListedColormap(['pink', 'lightblue','lightgreen'])
cmap_bold = ListedColormap(['red', 'darkblue','darkgreen'])


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

xs = df['Al']
ys = df['Co']
zs = df['Cr']
y = df["ID"]

plt.scatter(ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w'), y, cmap = cmap_bold, data = df)

ax.set_xlabel('at% Al', **tnrfont)
ax.set_ylabel('at% Co', **tnrfont)
ax.set_zlabel('at% Cr', **tnrfont)
ax.set_title("Visualization of 3 Continuous, Numeric Features (Al, Co, Cr)", **tnrfont)

'''

# Visualizing 4-D mix data using scatter plots
# leveraging the concepts of hue and depth

'''
fig = plt.figure(figsize=(8, 6))
t = fig.suptitle('AL - Co  - Cr - ID', fontsize=14)
ax = fig.add_subplot(111, projection='3d')

xs = list(df[['Al', 'ID']])
ys = list(df[['Co', 'ID']])
zs = list(df[['Cr', 'ID']])
#cs = list(df["ID"])
data_points = [(x, y, z) for x, y, z in zip(xs, ys, zs)]

for data in zip(data_points):
    if df["ID"].any() == 0.0:
        color = "red"
    elif df["ID"].any() == 1.0:
        color = "blue"
    elif df["ID"].any() == 2.0:
        color = "green"
    else:
        color = "yellow"
    
for data, color in zip(data_points, color):
   x, y, z = data
   ax.scatter(x, y, z, alpha=0.4, c=color, s=30)



ax.set_xlabel('at% Al')
ax.set_ylabel('at% Co')
ax.set_zlabel('at% Cr')

'''

fig = plt.figure(figsize=(8, 6))
t = fig.suptitle('AL - Co  - Cr - ID', fontsize=14)
ax = fig.add_subplot(111, projection='3d')

xs = np.array(df[['Al']])
ys = np.array(df[['Co']])
zs = np.array(df[['Cr']])


data_points = [(x, y, z) for x, y, z in zip(xs, ys, zs)]


def color():
    colors = []
    colorz = []
    
    for color in np.array(df["color"]):
        if df["ID"].any() == 0.0:
            color = ["red"]
            colorz = colors + color
            colorz += data
            
        elif df["ID"].any() == 1.0:
            color = ["blue"]
            colorz += data
            colorz = colors + color
            
        elif df["ID"].any() == 2.0:
            color = ["green"]
            colorz = colors + color
            colorz += data
            
        else:
            color = ["yellow"]
            colorz = colors + color
            colorz += data
    
        return colorz

color()
    
for data, colors in zip(data_points, colorz):
   x, y, z = data
   ax.scatter(x, y, z, alpha=0.4, c=colorz, s=30)



ax.set_xlabel('at% Al')
ax.set_ylabel('at% Co')
ax.set_zlabel('at% Cr')


''' just need to figure out how to make the function color() iterative)'''













'''
pycaret
https://www.youtube.com/watch?v=4Rn4YMLUjGc

'''







'''

#Plotting KNN

#take the first two features
X = features
y = hardness_class

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
plt.xlabel("at% Ni", **tnrfont)
plt.ylabel("at% Fe", **tnrfont)
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
clf = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, weights = "uniform")
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
plt.xlabel('at% Ni', **tnrfont)
plt.ylabel('at% Fe', **tnrfont)
plt.show()

''' '''Test data points are included in the scatter plot'''
'''
#Predict random composition
Ni = 42
Fe = 5
dataClass = clf.predict([[Ni,Fe]])
print('Prediction: ')

if dataClass == 0:
    print('Average, pink')
elif dataClass == 1:
    print('Soft, blue')
else:
    print('Hard, green')


'''
