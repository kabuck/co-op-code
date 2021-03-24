#Import necessary packages/libraries

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


#Checks amount of samples within certain range of HV
df.loc[df["Hardness"] <= (774), "ID"] = "Very Low HV"
df.loc[df["Hardness"] < 775.0, "ID"] = "Low HV"
df.loc[df["Hardness"] >= 775.0, 'ID'] = "Average HV"
df.loc[df["Hardness"] >= (775.0 + 77.5), 'ID'] = "High HV"

df["ID"].value_counts()



#Check for missing data

missing_data = df.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")
    
    
#Simple Linear Regression

#Scatter Plots

#at% Al vs. Hardness
plt.scatter(df.Al, df.Hardness, color = "red")
plt.xlabel("at% Al")
plt.ylabel("HV")
plt.title("at% Al vs. HV")
plt.show()
#at% Co vs. Hardness
plt.scatter(df.Co, df.Hardness, color = "orange")
plt.xlabel("at% Co")
plt.ylabel("HV")
plt.title("at% Co vs. HV")
plt.show()
#at% Cr vs. Hardness
plt.scatter(df.Cr, df.Hardness, color = "yellow")
plt.xlabel("at% Cr")
plt.ylabel("HV")
plt.title("at% Cr vs. HV")
plt.show()
#at% Cu vs. Predictd Hardness-Mean
plt.scatter(df.Cu, df.Hardness, color = "green")
plt.xlabel("at% Cu")
plt.ylabel("HV")
plt.title("at% Cu vs. HV")
plt.show()
#at% Fe vs. Hardness 
plt.scatter(df.Fe, df.Hardness, color = "blue")
plt.xlabel("at% Fe")
plt.ylabel("HV")
plt.title("at% Fe vs. HV")
plt.show()
#at% Ni vs. Hardness
plt.scatter(df.Ni, df.Hardness, color = "purple")
plt.xlabel("at% Ni")
plt.ylabel("HV")
plt.title("at% Ni vs. HV")
plt.show()

#Histograms (y axis = number of samples; x axis = atomic %)

#at% Al vs. Hardness
plothist = df[["Al"]]
plothist.hist(color = "red")
plt.ylabel("Number of Samples")
plt.xlabel("at% Al")
plt.title("Amount of Samples with at% Al")

plt.show()
#at% Co vs. Hardness
plothist = df[["Co"]]
plothist.hist(color = "orange")
plt.ylabel("Number of Samples")
plt.xlabel("at% Co")
plt.title("Amount of Samples with at% Co")

plt.show()
#at% Cr vs. Hardness
plothist = df[["Cr"]]
plothist.hist(color = "yellow")
plt.ylabel("Number of Samples")
plt.xlabel("at% Cr")
plt.title("Amount of Samples with at% Cr")

plt.show()
#at% Cu vs. Predictd Hardness-Mean
plothist = df[["Cu"]]
plothist.hist(color = "green")
plt.ylabel("Number of Samples")
plt.xlabel("at% Cu")
plt.title("Amount of Samples with at% Cu")

plt.show()
#at% Fe vs. Hardness
plothist = df[["Fe"]]
plothist.hist(color = "blue")
plt.ylabel("Number of Samples")
plt.xlabel("at% Fe")
plt.title("Amount of Samples with at% Fe")

plt.show()
#at% Ni vs. Hardness
plothist = df[["Ni"]]
plothist.hist(color = "purple")
plt.ylabel("Number of Samples")
plt.xlabel("at% Ni")
plt.title("Amount of Samples with at% Ni")

plt.show()


#Creating a test/training set (Al)

from sklearn import linear_model
samples = np.random.rand(len(df))<0.8
train = df[samples] #80% of data used to train the model
test = df[~samples] #remainder of data used to test the model

#Train Data Distribution

plt.scatter(train.Al, train.Hardness, color="pink")
plt.xlabel("at% Al")
plt.ylabel("Predicted HV")
plt.title("at% Al vs. HV Training Data")

plt.show() #prints the raw data

regrAl = linear_model.LinearRegression() 
train_x = np.asanyarray(train[["Al"]]) #Converts the training data (IV) to an array
train_y = np.asanyarray(train[["Hardness"]]) #Converts training data (DV) to an array
regrAl.fit(train_x, train_y) #fits a line of best fit to the data

print("Coefficients: ", regrAl.coef_) #Slope of the regression line
print("Intercept: ", regrAl.intercept_) #Intercept of the regression line

plt.scatter(train.Al, train.Hardness, color = "pink", alpha = 0.5) #plots trianing data
plt.plot(train_x, regrAl.coef_[0][0]*train_x + regrAl.intercept_[0], "-r") #plots the regression line
plt.xlabel("at% Al")
plt.ylabel("Prediced HV")
plt.title("Simple Linear Regression Model: at% Al vs. HV")

plt.show()

from sklearn.metrics import r2_score 

test_x = np.asanyarray(test[["Al"]]) #Converts testing data (IV) to an array
test_y = np.asanyarray(test[["Hardness"]]) #Converts testing data (DV) to an array
test_Al = regrAl.predict(test_x) #Predicts the HV of the testing values using the regression equation

#Accuracy Metrics
print("Mean Absolute Error: %.2f"%np.mean(np.absolute(test_Al-test_y)))
print("Residual Mean Squared Error: %.2f"%np.mean((test_Al-test_y)**2))
print("R2 Score: %.2f"%r2_score(test_y, test_Al))


#Creating a test/training set (Cu)

from sklearn import linear_model
samples = np.random.rand(len(df))<0.8
train = df[samples]
test = df[~samples]

#Train Data Distribution

plt.scatter(train.Cu, train.Hardness, color="LightGreen")
plt.xlabel("at% Cu")
plt.ylabel("Hardness, HV")
plt.title("at% Cu vs. HV Training Data")
plt.show()

regrCu = linear_model.LinearRegression()
train_x_Cu = np.asanyarray(train[["Cu"]])
train_y_Cu = np.asanyarray(train[["Hardness"]])
regrCu.fit(train_x_Cu, train_y_Cu)

print("Coefficients: ", regrCu.coef_)
print("Intercept: ", regrCu.intercept_)

plt.scatter(train.Cu, train.Hardness, color = "LightGreen")
plt.plot(train_x_Cu, regrCu.coef_[0][0]*train_x_Cu + regrCu.intercept_[0], color = "Green")
plt.xlabel("at% Cu")
plt.ylabel("Hardness, HV")
plt.title("Simple Linear Regression Model: at% Cu vs. HV")

plt.show()

test_x_Cu = np.asanyarray(test[["Cu"]]) #Converts testing data (IV) to an array
test_y_Cu = np.asanyarray(test[["Hardness"]]) #Converts testing data (DV) to an array
test_Cu = regrCu.predict(test_x_Cu)
#Accuracy Metrics
print("Mean Absolute Error: %.2f"%np.mean(np.absolute(test_Cu-test_y_Cu)))
print("Residual Mean Squared Error: %.2f"%np.mean((test_Cu-test_y_Cu)**2))
print("R2 Score: %.2f"%r2_score(test_y_Cu, test_Cu))


#SLR with Normalization
NAlloyNo = np.arange(0, 155, 1)
Nregrline = regrCu.intercept_ + regrCu.coef_[0][0]*NAlloyNo
NregrCuy = np.array(Nregrline)
Nregrline2 = (NregrCuy - NregrCuy.min())/(NregrCuy.max() - NregrCuy.min())
NregrCunorm = np.array(Nregrline2)
Npredyval = np.array(NregrCunorm)

Nx = np.arange(0, 155, 1)
Ny = np.array((df[["Hardness"]] - df[["Hardness"]].min())/(df[["Hardness"]].max() - df[["Hardness"]].min()))

plt.scatter(Nx, Ny, color="LightGreen")
plt.plot(NAlloyNo, Npredyval, "DarkGreen")
plt.xlabel("Alloy Number")
plt.ylabel("Normalized Hardness, HV")
plt.title("Normalized SLR Model: at% Cu vs. HV")

plt.show()

Ntest_x_Cu = np.asanyarray(test[["Cu"]])
Ntest_y_Cu = np.asanyarray(test[["Hardness"]])
Ntest_Cu = regrCu.predict(test_x)

print("MAE: %.2f"%np.mean(np.absolute(Ntest_Cu-Ntest_y_Cu)))
print("Residual MSE: %.2f"%np.mean((Ntest_Cu-Ntest_y_Cu)**2))
print("R2 Score: %.2f"%r2_score(Ntest_y_Cu, Ntest_Cu))

#SLR with Standardization
SAlloyNo = np.arange(0, 155, 1)
Sregrline1 = regrCu.intercept_[0] + regrCu.coef_[0][0]*SAlloyNo
SregrCuy2 = np.array(Sregrline1)
Sregrline2 = (SregrCuy2 - SregrCuy2.mean())/(SregrCuy2.std())
Sstdyvalues = np.array(Sregrline2)
Spredyval2 = np.array(Sstdyvalues)

#standardization of scatter plot data
Sx = np.arange(0, 155, 1)
Sy = np.array((df[["Hardness"]] - df[["Hardness"]].mean())/(df[["Hardness"]].std()))

plt.scatter(Sx, Sy, color="LightGreen")
plt.plot(SAlloyNo, Spredyval2, "DarkGreen")
plt.xlabel("Alloy Number")
plt.ylabel("Standardized Hardness, HV")
plt.title("Standardized SLR Model: at% Cu vs. HV")

plt.show()

from sklearn.metrics import r2_score

Stest_x_Cu = np.asanyarray(test[["Cu"]])
Stest_y_Cu = np.asanyarray(test[["Hardness"]])
Stest_Cu = regrCu.predict(Stest_x_Cu)

print("MAE: %.2f"%np.mean(np.absolute(Stest_Cu-Stest_y_Cu)))
print("Residual MSE: %.2f"%np.mean((Stest_Cu-Stest_y_Cu)**2))
print("R2 Score: %.2f"%r2_score(Stest_y_Cu, test_Cu))

#At this point, it's clear that single linear regression is not the best model for this data.
#Multiple linear regression will likely not be a good fit either, but we'll test it out just in case.



#Multiple Nonlinear Regression

from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[["Al", "Co", "Cr", "Cu", "Fe", "Ni"]])
y = np.asanyarray(train[["Hardness"]])
regr.fit(x,y)
print("coefficients: ", regr.coef_)

yhat = regr.predict(test[["Al", "Co", "Cr", "Cu", "Fe", "Ni"]])
X = np.asanyarray(test[["Al", "Co", "Cr", "Cu", "Fe", "Ni"]])
Y = np.asanyarray(test[["Hardness"]])

print("Residual MSE: %.2f"%np.mean((yhat-Y)**2))
print("variance score: %.2f"%regr.score(X,Y))

#normalization of regression curve
xx = np.arange(0, 155, 0.1)
eqn = regr.intercept_[0] + regr.coef_[0][0] + regr.coef_[0][1]*xx + regr.coef_[0][2]*np.power(xx, 2) + regr.coef_[0][3]*np.power(xx, 3) + regr.coef_[0][4]*np.power(xx, 4) + regr.coef_[0][5]*np.power(xx, 5)
yvalues = np.array(eqn)
eqn2 = (yvalues - yvalues.min())/(yvalues.max()-yvalues.min())
normyvalues = np.array(eqn2)
yy = np.array(normyvalues)

#normalization of scatter plot data
x = np.arange(0, 155, 1)
y = np.array((df[["Hardness"]] - df[["Hardness"]].min())/(df[["Hardness"]].max() - df[["Hardness"]].min()))

plt.scatter(x, y, color="silver")
plt.plot(xx, yy, "black")
plt.xlabel("Alloy Number")
plt.ylabel("Normalized Hardness, HV")
plt.title("MLR Normalized Alloy No. vs. HV")

plt.show()


#standardization of regression curve
xx = np.arange(0, 155, 0.1)
eqn = regr.intercept_[0] + regr.coef_[0][0] + regr.coef_[0][1]*xx + regr.coef_[0][2]*np.power(xx, 2) + regr.coef_[0][3]*np.power(xx, 3) + regr.coef_[0][4]*np.power(xx, 4) + regr.coef_[0][5]*np.power(xx, 5)
yvalues = np.array(eqn)
eqn2 = (yvalues - yvalues.mean())/(yvalues.std())
stdyvalues = np.array(eqn2)
yy = np.array(stdyvalues)

#standardization of scatter plot data
x = np.arange(0, 155, 1)
y = np.array((df[["Hardness"]] - df[["Hardness"]].mean())/(df[["Hardness"]].std()))

plt.scatter(x, y, color="grey")
plt.plot(xx, yy, "black")
plt.xlabel("Alloy Number")
plt.ylabel("Standardized Hardness, HV")
plt.title("MLR Standardized Alloy No. vs. HV")

plt.show()

#Standardization of data results in a better regression curve fit (visually)


#Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

train_x = np.asanyarray(train[["Al"]])
train_y = np.asanyarray(train[["Hardness"]])

poly = PolynomialFeatures(degree = 2)
train_x_poly = poly.fit_transform(train_x)
train_x_poly

polyregr = linear_model.LinearRegression()
train_y_ = polyregr.fit(train_x_poly, train_y)
print("Coefficients: ", polyregr.coef_)
print("Intercept: ", polyregr.intercept_)

plt.scatter(train.Al, train.Hardness, color = "pink", alpha = 0.5)
XX = np.arange(0, 50, 0.1)
yy = polyregr.intercept_[0] + polyregr.coef_[0][1]*XX + polyregr.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, "-r")
plt.xlabel("at% Al")
plt.ylabel("Hardness, HV")
plt.title("Polynomial Regression: at% Al vs. HV")

plt.show()

from sklearn.metrics import r2_score
test_x_poly = poly.fit_transform(test_x)
test_y_ = polyregr.predict(test_x_poly)
print("MAE: %.2f"%np.mean(np.absolute(test_y_-test_y)))
print("Residual MSE: %.2f"%np.mean((test_y_-test_y)**2))
print("R2 score: %.2f"%r2_score(test_y, test_y_))

#Not the best model







'''
y = np.array(y)
y = y.reshape(-1, 1)
y_scaled = scaler.fit_transform(y)
y = pd.DataFrame(y_scaled)
y.columns = ["Hardness"]

print(y.head(5))
'''