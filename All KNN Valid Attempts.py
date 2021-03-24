#KNN Attempt 1

import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
from sklearn import linear_model

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

#Segregating variables

x = df.drop(["Hardness"], axis = 1)
y = df["Hardness"]
print(x.shape, "\n", y.shape)

#Scaling the data


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
x = pd.DataFrame(x_scaled, columns = x.columns)

#y_scaled = scaler.fit_transform(y)
#y = pd.DataFrame(y_scaled, columns = y.columns)

#divide into test/train data

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, random_state = 56, stratify = y)
train_y = train_y.reshape(-1,1)
train_x = train_x.reshape(1,-1)
test_x = test_x.reshape(1, -1)

#Implement KNN

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import f1_score

k = 5
clf = KNN(n_neighbors = k)
clf.fit(train_x, train_y)
test_predict = clf.predict([[test_x]])
f1score = f1_score(test_predict, test_y)

print("Test F1 Score: ", f1score)

#Elbow for classfier

def Elbow(K):
    test_error = []
    
    for i in K:
        clf = KNN(n_neighbors = i)
        clf.fit(train_x, train_y)
        tmp = clf.predict(test_x)
        tmp = f1_score(tmp, test_y)
        error = 1-tmp
        test_error.append(error)
    
    return test_error

K = range(1, 20, 1)

test = Elbow(K)

plt.plot(k, test)
plt.xlabel("K Neighbors")
plt.ylabel("Test Error")
plt.title("Elbow Curve for Test")

#train the model again

clf = KNN(n_neighbors = 10)
clf.fit(train_x, train_y)
test_predict = clf.predict(test_x)
f1score = f1_score(test_predict, test_y)
print("Test F1 Score: ", f1score)


#KNN Regression

#Import/Download Data

df = pd.read_csv("C:/Users/Matt/Documents/1 Kayla's Stuff/Week 4 Materials/ds1.csv")
print(df.shape)
df.head(5)

#Format the data

headers = ["Alloy No.", "Al", "Co", "Cr", "Cu", "Fe", "Ni", "Hardness"]
df.columns = headers #Changes the headers
df.drop([0,0], inplace=True) #removes the extra headers
print(df.head(5))


#Segregating variables

x = df.drop(["Hardness"], axis = 1)
x = df.drop(["Alloy No."], axis = 1)
print(x.head(5))


y = df
y = y.drop(["Al"], axis = 1)
y = y.drop(["Cu"], axis = 1)
y = y.drop(["Cr"], axis = 1)
y = y.drop(["Co"], axis = 1)
y = y.drop(["Fe"], axis = 1)
y = y.drop(["Ni"], axis = 1)
print(y.head(5))



print(x.shape, "\n", y.shape)

#Scaling the data

'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
x = pd.DataFrame(x_scaled, columns = x.columns)
print(x)

y = np.array(y)
y = y.reshape(-1, 1)
y_scaled = scaler.fit_transform(y)
y = pd.DataFrame(y_scaled)
y.columns = ["Hardness"]

print(y_scaled)

'''

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x,y)
'''
train_y = train_y.reshape(-1,1)
train_x = train_x.reshape(-1,1)
test_x = test_x.reshape(-1,1)
print(test_x, "\n", test_y)
'''


from sklearn.neighbors import KNeighborsRegressor as KNNR
from sklearn.metrics import mean_squared_error as mse

regr = KNNR(n_neighbors = 5)
regr.fit(train_x, train_y)

test_predict = regr.predict(test_x)
MSE = mse(test_predict, test_y)
print("MSE: ", MSE)



#Elbow for classifier
def Elbow(K):
    test_mse = []
    
    for i in K:
        regr = KNNR(n_neighbors = i)
        regr.fit(train_x, train_y)
        tmp1 = regr.predict(test_x)
        tmp1 = mse(tmp1, test_y)
        test_mse.append(tmp1)
        
    return test_mse

K = range(1,15)

test = Elbow(K)
plt.plot(K, test)
plt.xlabel("K Neighbors")
plt.ylabel("Test MSE")
plt.title("Elbow Curve for Test")
plt.show()

regr = KNNR(n_neighbors = 8)
regr.fit(train_x, train_y)
test_predict = regr.predict(test_x)
MSE = mse(test_predict, test_y)
print("Test MSE: ", MSE)



#KNN Attempt 2

import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
from sklearn import linear_model

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

#Segregating variables

x = df.drop(["Hardness"], axis = 1)
y = df["Hardness"]
print(x.shape, "\n", y.shape)

#Scaling the data


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
x = pd.DataFrame(x_scaled, columns = x.columns)

y_scaled = scaler.fit_transform(y)
y = pd.DataFrame(y_scaled, columns = y.columns)


#divide into test/train data

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, random_state = 56, stratify = y)
train_y = train_y.reshape(-1,1)
train_x = train_x.reshape(1,-1)
test_x = test_x.reshape(1, -1)

#Implement KNN

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import f1_score

k = 5
clf = KNN(n_neighbors = k)
clf.fit(train_x, train_y)
test_predict = clf.predict([[test_x]])
f1score = f1_score(test_predict, test_y)

print("Test F1 Score: ", f1score)

#Elbow for classfier

def Elbow(K):
    test_error = []
    
    for i in K:
        clf = KNN(n_neighbors = i)
        clf.fit(train_x, train_y)
        tmp = clf.predict(test_x)
        tmp = f1_score(tmp, test_y)
        error = 1-tmp
        test_error.append(error)
    
    return test_error

K = range(1, 20, 1)

test = Elbow(K)

plt.plot(k, test)
plt.xlabel("K Neighbors")
plt.ylabel("Test Error")
plt.title("Elbow Curve for Test")

#train the model again

clf = KNN(n_neighbors = 10)
clf.fit(train_x, train_y)
test_predict = clf.predict(test_x)
f1score = f1_score(test_predict, test_y)
print("Test F1 Score: ", f1score)







#KNN Regression

#Import/Download Data

df = pd.read_csv("C:/Users/Matt/Documents/1 Kayla's Stuff/Week 4 Materials/ds1.csv")
print(df.shape)
df.head(5)

#Format the data

headers = ["Alloy No.", "Al", "Co", "Cr", "Cu", "Fe", "Ni", "Hardness"]
df.columns = headers #Changes the headers
df.drop([0,0], inplace=True) #removes the extra headers
print(df.head(5))


#Segregating variables

x = df["Al"]
y = df["Cu"]

import seaborn as sns
sns.set(font = "Times New Roman", style = "whitegrid", font_scale = .5)
sns.axes_style("ticks", {"xtick.major.size": 20, "ytick.major.size": 20})
plt.scatter(x,y)
plt.gcf().set_size_inches(5, 5)


#Scaling the data


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
x = pd.DataFrame(x_scaled, columns = x.columns)
print(x)

plt.plot(x,y)
plt.show()

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x,y)

train_y = train_y.stack()
train_x = train_x.stack()
test_x = test_x.stack()

print(train_x, "\n", train_y)


from sklearn.neighbors import KNeighborsRegressor as KNNR
from sklearn.metrics import mean_squared_error as mse

regr = KNNR(n_neighbors = 5)
regr.fit(train_x, train_y)

test_predict = regr.predict(test_x)
MSE = mse(test_predict, test_y)
print("MSE: ", MSE)



#Elbow for classifier
def Elbow(K):
    test_mse = []
    
    for i in K:
        regr = KNNR(n_neighbors = i)
        regr.fit(train_x, train_y)
        tmp1 = regr.predict(test_x)
        tmp1 = mse(tmp1, test_y)
        test_mse.append(tmp1)
        
    return test_mse

K = range(1,10)

test = Elbow(K)
plt.plot(K, test)
plt.xlabel("K Neighbors")
plt.ylabel("Test MSE")
plt.title("Elbow Curve for Test")
plt.show()

regr = KNNR(n_neighbors = 4)
regr.fit(train_x, train_y)
test_predict = regr.predict(test_x)
MSE = mse(test_predict, test_y)
print("Test MSE: ", MSE)


#KNN Attempt 3

import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
from sklearn import linear_model

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


print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 15

# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ['darkorange', 'c', 'darkblue']

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=iris.target_names[y],
                    palette=cmap_bold, alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])

plt.show()



#KNN Attempt 4

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#Import/format data
df = pd.read_csv("C:/Users/Matt/Documents/1 Kayla's Stuff/Week 4 Materials/ds1.csv")
print(df.shape)
df.head(5)

headers = ["Alloy No.", "Al", "Co", "Cr", "Cu", "Fe", "Ni", "Hardness"]
df.columns = headers #Changes the headers
df.drop([0,0], inplace=True) #removes the extra headers
print(df.head(5))
df = df[["Al", "Co", 'Cr', 'Cu', "Fe", "Ni", "Hardness"]].astype("float")
df.info()
print(df.min)
df.head(5)

df["Hardness"].value_counts()
df.columns

X = df[["Al", "Co", 'Cr', 'Cu', "Fe", "Ni"]].values.astype(float)
X[0:5]

y = df["Hardness"].values
y[0:5]

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 4)
print("Train set:", X_train.shape, y_train.shape)
print("Test set: ", X_test.shape, y_test.shape)

#With K = 1
k = 1

neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
neigh

yhat = neigh.predict(X_test)
yhat[0:5]

#Accuracy evaluation

print("1 Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("1 Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))



Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
for n in range(1, Ks):
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
mean_acc


#KNN Attempt 5

#Tried to do this: https://github.com/shenghuanjie/CS189-2018Spring-HW/blob/787db97a0fd9f72e28e7489df4613c7861f83b86/HW11/hw11-data/world_values_starter.py


#KNN

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

#Import/Download Data

df = pd.read_csv("C:/Users/Matt/Documents/1 Kayla's Stuff/Week 4 Materials/exp_data.csv")
print(df.shape)


#Format the data

headers = ["Alloy No.", "Al", "Co", "Cr", "Cu", "Fe", "Ni", "PredictedHVMean", "PredictedHVsd"]
df.columns = headers #Changes the headers
df.drop([0,0], inplace=True) #removes the extra headers
print(df.head(5))
df = df[["Al", "Co", 'Cr', 'Cu', "Fe", "Ni", "PredictedHVMean", "PredictedHVsd"]].astype("float")
df.info()

#Add classification column

df.loc[df["PredictedHVMean"] <= (775.0 - 77.5), "ID"] = "Very Low HV"
df.loc[df["PredictedHVMean"] < 775.0, "ID"] = "Low HV"
df.loc[df["PredictedHVMean"] >= 775.0, 'ID'] = "Average HV"
df.loc[df["PredictedHVMean"] >= (775.0 + 77.5), 'ID'] = "High HV"

df["ID"].value_counts()


from math import sqrt
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from world_values_utils import import_world_values_data
from world_values_utils import hdi_classification
from world_values_utils import calculate_correlations
from world_values_utils import plot_pca
from world_values_utils import print

from world_values_pipelines import ridge_regression_pipeline
from world_values_pipelines import lasso_regression_pipeline
from world_values_pipelines import k_nearest_neighbors_regression_pipeline
from world_values_pipelines import svm_classification_pipeline
from world_values_pipelines import k_nearest_neighbors_classification_pipeline
from world_values_pipelines import tree_classification_pipeline

from world_values_parameters import regression_ridge_parameters
from world_values_parameters import regression_lasso_parameters
from world_values_parameters import regression_knn_parameters
from world_values_parameters import classification_svm_parameters
from world_values_parameters import classification_knn_parameters
from world_values_parameters import classification_tree_parameters

import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from world_values_parameters import regression_knn_parameters_weighted
from world_values_pipelines import k_nearest_neighbors_regression_pipeline_scaled
from world_values_pipelines import k_nearest_neighbors_regression_pipeline_minmax
from world_values_pipelines import k_nearest_neighbors_regression_pipeline_binary
from world_values_pipelines import svm_classification_pipeline_pca_scaled
from world_values_parameters import classification_svm_parameters_pca_scaled
from world_values_parameters import classification_svm_parameters_rbf
from world_values_pipelines import k_nearest_neighbors_classification_pipeline_scaled




def main():
    values_train, HV_train, values_test = df[["Al", "Co", "Cr", "Cu", "Fe", "Ni", "PredictedHVMean"]]
    
    HV_scaler = StandardScaler(with_std=False)
    HV_shifted_train = HV_scaler.fit_transform(HV_train)
    
    HV_class_train = HV_train["PredictedHVMean"].apply(HV_class)
    
    calculate_correlations(values_train, HV_train)
    
    plot_pca(values_train, HV_train, HV_class_train)
    
    regression_grid_searches(training_features=values_train, training_labels=HV_train, title="Figure_1")
    
    neigh = NearestNeighbors(n_neighbors=6).fit(values_train)
    distances, indices = neigh.kneighbors([values_train.iloc[50]])
    cpds_train = df
    cpds_train = df["Alloy No."]
    
    _rmse_grid_search(training_features=values_train, training_labels=HV_train, pipeline=k_nearest_neighbors_regression_pipeline, parameters=regression_knn_parameters_weighted, technique="knn", title="Figure_2")
    
    stdscaler = StandardScaler()
    stdscaler.fit(values_train)
    values_train = stdscaler.transform(values_train)
    neigh = NearestNeighbors(n_neighbors=6).fit(values_train)
    distances, indices = neigh.kneighbors([values_train[50, :]])
    cpds_train = df
    cpds_train = df["Alloy No."]
    
    
    _rmse_grid_search(training_features=values_train, training_labels=HV_train, pipeline=k_nearest_neighbors_regression_pipeline_minmax, parameters=regression_knn_parameters_weighted, technique='knn', title='Figure_3')
    print("k Nearest Neighbors Regression (Binarizer)")
    _rmse_grid_search(training_features=values_train, training_labels=HV_train, pipeline=k_nearest_neighbors_regression_pipeline_binary, parameters=regression_knn_parameters_weighted, technique='knn', title='Figure_4')
    
    print('The current best model is: ')
    grid = _rmse_grid_search(training_features=values_train, training_labels=HV_train, pipeline=k_nearest_neighbors_regression_pipeline_scaled, parameters=regression_knn_parameters_weighted, technique='knn', title='Figure_5')
    pred_values = grid.predict(values_test)
    cpds_test = df
    cpds_test = df[['Alloy No.']]
    pred_values = pred_values.flatten()
    cpds_test['Alloy No.'] = pred_values
    print(cpds_test)   
    
    _accuracy_grid_search(values_train, HV_class_train, svm_classification_pipeline, classification_svm_parameters)
    
    _accuracy_grid_search(values_train, HV_class_train, svm_classification_pipeline_pca_scaled, classification_svm_paramters_pca_scaled)
    
    _accuracy_grid_search(values_train, HV_class_train, svm_classification_pipeline, classification_svm_parameters_rbf)
    
    _accuracy_grid_search(values_train, HV_class_train, k_nearest_neighbors_classification_pipeline, classification_knn_parameters)
    
    _accuracy_grid_search(values_train, HV_class_train, k_nearest_neighbors_classification_pipeline_scaled, classification_knn_parameters)
    
    values_test = df["PredicedHVMean"]
    grid = _rmse_grid_search(training_features=values_train, training_labels=HV_train, pipeline=k_nearest_neighbors_regression_pipeline_scaled, parameters=regression_knn_parameters_weighted, technique='knn', title='Figure_6')
    pred_values = grid.predict(values_test)
    cpds_test = df
    cpds_test = df[['Alloy No.']]
    pred_values = pred_values.flatten()
    cpds_test['Alloy No.'] = pred_values
    print(cpds_test)
    
    classification_grid_searches(training_features=values_train, training_classes=HV_class_train)
    
    
    
#KNN Attempt 6
    

#skeleton code from github: https://github.com/rampk/imputation-analysis/blob/c3c592f4a8badd4364b49a507019b7a7b269dc23/Analysis_program/visualize_metrics.py
#not quite sure what to change to get it to do what I want


import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_performance(values, title, folder):
    df = pd.read_csv("C:/Users/Matt/Documents/1 Kayla's Stuff/Week 4 Materials/exp_data.csv")
    xticks = ['Mean', 'Median', 'Mode']
    plt.boxplot(values)
    plt.violinplot(values)
    plt.xticks(range(0, 50), xticks)
    plt.title("HV")
    plt.savefig(df)
    plt.clf()


def visualize_performance(metrics, inputs, writer):
    # Create a directory for storing images
    run_by = inputs['run_by']
    dir_name = f'../Results/Images/{run_by}_images_{writer.current_num}'
    os.makedirs(dir_name)

    # Visualize the metrics
    impute_types = [metrics.mean, metrics.median, metrics.mode]

    # Visualize Accuracy
    forest = []
    logistic = []
    knn = []
    svm = []

    for impute in impute_types:
        for algorithm in impute:
            if algorithm == 'RandomForest':
                forest.append(impute[algorithm]['Accuracy'])
            elif algorithm == 'LogisticRegression':
                logistic.append(impute[algorithm]['Accuracy'])
            elif algorithm == 'KNeighbors':
                knn.append(impute[algorithm]['Accuracy'])
            else:
                svm.append(impute[algorithm]['Accuracy'])

    plot_performance(forest, "RandomForest_Accuracy", dir_name)
    plot_performance(logistic, "LogisticRegression_Accuracy", dir_name)
    plot_performance(knn, "KNN_Accuracy", dir_name)
    plot_performance(svm, "SVM_Accuracy", dir_name)

    # Visualize AUC
    forest = []
    logistic = []
    knn = []
    svm = []

    for impute in impute_types:
        for algorithm in impute:
            if algorithm == 'RandomForest':
                forest.append(impute[algorithm]['AUC'])
            elif algorithm == 'LogisticRegression':
                logistic.append(impute[algorithm]['AUC'])
            elif algorithm == 'KNeighbors':
                knn.append(impute[algorithm]['AUC'])
            else:
                svm.append(impute[algorithm]['AUC'])

    plot_performance(forest, "RandomForest_AUC", dir_name)
    plot_performance(logistic, "LogisticRegression_AUC", dir_name)
    plot_performance(knn, "KNN_AUC", dir_name)
    plot_performance(svm, "SVM_AUC", dir_name)

    # Visualize Precision
    forest = []
    logistic = []
    knn = []
    svm = []

    for impute in impute_types:
        for algorithm in impute:
            if algorithm == 'RandomForest':
                forest.append(impute[algorithm]['Precision'])
            elif algorithm == 'LogisticRegression':
                logistic.append(impute[algorithm]['Precision'])
            elif algorithm == 'KNeighbors':
                knn.append(impute[algorithm]['Precision'])
            else:
                svm.append(impute[algorithm]['Precision'])

    plot_performance(forest, "RandomForest_Precision", dir_name)
    plot_performance(logistic, "LogisticRegression_Precision", dir_name)
    plot_performance(knn, "KNN_Precision", dir_name)
    plot_performance(svm, "SVM_Precision", dir_name)

    # Visualize Recall
    forest = []
    logistic = []
    knn = []
    svm = []

    for impute in impute_types:
        for algorithm in impute:
            if algorithm == 'RandomForest':
                forest.append(impute[algorithm]['Recall'])
            elif algorithm == 'LogisticRegression':
                logistic.append(impute[algorithm]['Recall'])
            elif algorithm == 'KNeighbors':
                knn.append(impute[algorithm]['Recall'])
            else:
                svm.append(impute[algorithm]['Recall'])

    plot_performance(forest, "RandomForest_Recall", dir_name)
    plot_performance(logistic, "LogisticRegression_Recall", dir_name)
    plot_performance(knn, "KNN_Recall", dir_name)
    plot_performance(svm, "SVM_Recall", dir_name)

    # Visualize F1
    forest = []
    logistic = []
    knn = []
    svm = []

    for impute in impute_types:
        for algorithm in impute:
            if algorithm == 'RandomForest':
                forest.append(impute[algorithm]['F1'])
            elif algorithm == 'LogisticRegression':
                logistic.append(impute[algorithm]['F1'])
            elif algorithm == 'KNeighbors':
                knn.append(impute[algorithm]['F1'])
            else:
                svm.append(impute[algorithm]['F1'])

    plot_performance(forest, "RandomForest_F1", dir_name)
    plot_performance(logistic, "LogisticRegression_F1", dir_name)
    plot_performance(knn, "KNN_F1", dir_name)
    plot_performance(svm, "SVM_F1", dir_name)
    






