#violin plots

#Import libraries
import numpy as np 
import matplotlib.pyplot as plt 
from random import randint 
import seaborn as sns
import pandas as pd
import scipy as sp
import matplotlib.ticker as ticker
import pylab as pl
'''
import plotly.express as px
'''
#Import/Download Data

df = pd.read_csv("C:/Users/Matt/Documents/1 Kayla's Stuff/Week 4 Materials/ds1.csv")
print(df.shape) #returns (Row No., Column No.)
print(df.head(5)) #Prints first 5 rows of the dataframe


#Format the data

headers = ["Alloy No.", "Al", "Co", "Cr", "Cu", "Fe", "Ni", "Hardness"]
df.columns = headers #Changes the headers
df.drop([0,0], inplace=True) #removes the extra headers
df = df[["Al", "Co", 'Cr', 'Cu', "Fe", "Ni", "Hardness"]].astype("float")
df.info() #Checks that all data is in the right form
print(df.head(5))

#To change font/color for plots
tnrfont = {'fontname':'Times New Roman', 'color':'black'}

#Amount of each element at given at%

#Dropped Hardness column
dfalloys = df.drop(df.columns[6], axis=1)
dfalloys.head(5)

fig = plt.gcf()
fig.set_size_inches(8,10) #adjusts graph proportions
sns.set(font_scale = 1.5) 
sns.set(style = "white") #standard graph background
sns.violinplot(data = dfalloys) #makes the violin plot

#Add axis labels and a title:
plt.xlabel("Element", **tnrfont)
plt.ylabel("at%", **tnrfont)
plt.title("at% of Each Element for 155 samples", **tnrfont)

plt.show() #prints graph


#Hardness of Al
fig = plt.gcf()
fig.set_size_inches(8, 10)
sns.set(font_scale = 2)

#yticks = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

#fig, ax = plt.subplots() #little formatting change

sns.set(style = "whitegrid")
sns.violinplot(x = "Hardness", y = "Al", data = df, scale = "width", width = 1, orient = "h")
plt.ylabel("at% Al", **tnrfont)
plt.xlabel("Hardness, HV", **tnrfont)
plt.title("Relationship between at% Al and Hardness", **tnrfont)

plt.show()

#Hardness of Co
fig = plt.gcf()
fig.set_size_inches(8, 10)
sns.set(font_scale = 2)

sns.set(style = "whitegrid")
sns.violinplot(x = "Hardness", y = "Co", data = df, scale = "width", width = 1, orient = "h")
plt.ylabel("at% Co", **tnrfont)
plt.xlabel("Hardness, HV", **tnrfont)
plt.title("Relationship between at% Co and Hardness", **tnrfont)

plt.show()

#Hardness of Cr
fig = plt.gcf()
fig.set_size_inches(8, 10)
sns.set(font_scale = 2)

sns.set(style = "whitegrid")
sns.violinplot(x = "Hardness", y = "Cr", data = df, scale = "width", width = 1, orient = "h")
plt.ylabel("at% Cr", **tnrfont)
plt.xlabel("Hardness, HV", **tnrfont)
plt.title("Relationship between at% Cr and Hardness", **tnrfont)

plt.show()

#Hardness of Cu
fig = plt.gcf()
fig.set_size_inches(8, 10)
sns.set(font_scale = 2)

sns.set(style = "whitegrid")
sns.violinplot(x = "Hardness", y = "Cu", data = df, scale = "width", width = 1, orient = "h")
plt.ylabel("at% Cu", **tnrfont)
plt.xlabel("Hardness, HV", **tnrfont)
plt.title("Relationship between at% Cu and Hardness", **tnrfont)

plt.show()

#Hardness of Fe
fig = plt.gcf()
fig.set_size_inches(8, 10)
sns.set(font_scale = 2)

sns.set(style = "whitegrid")
sns.violinplot(x = "Hardness", y = "Fe", data = df, scale = "width", width = 1, orient = "h")
plt.ylabel("at% Fe", **tnrfont)
plt.xlabel("Hardness, HV", **tnrfont)
plt.title("Relationship between at% Fe and Hardness", **tnrfont)

plt.show()

#Hardness of Ni
fig = plt.gcf()
fig.set_size_inches(8, 10)
sns.set(font_scale = 2)

sns.set(style = "whitegrid")
sns.violinplot(x = "Hardness", y = "Ni", data = df, scale = "width", width = 1, orient = "h")
plt.ylabel("at% Ni", **tnrfont)
plt.xlabel("Hardness, HV", **tnrfont)
plt.title("Relationship between at% Ni and Hardness", **tnrfont)

plt.show()





'''

TRYING TO FIGURE OUT HOW TO
OVERLAP INDIVIDUAL PLOTS

'''



'''
xal = np.array(df["Al"])
xco = np.array(df["Co"])
xcr = np.array(df["Cr"])
xcu = np.array(df["Cu"])
xfe = np.array(df["Fe"])
xni = np.array(df["Ni"])
hardness = np.array(df["Hardness"])


import numpy as np 
import matplotlib.pyplot as plt 
from random import randint 
import seaborn as sns
import pandas as pd
import scipy as sp
import matplotlib.ticker as ticker























from matplotlib import pyplot as plt
df.head(5)

dfalloys = df.drop(df.columns[6], axis=1)
dfalloys.head(5)


Al = df["Al"]
Co = df["Co"]
Cr = df["Cr"]
Cu = df["Cu"]
Fe = df["Fe"]
Ni = df["Ni"]
ydata = (Al, Co, Cr, Cu, Fe, Ni)

fig = plt.gcf()
fig.set_size_inches(8,10)
sns.set(font_scale = 1.5)

fig, axes = plt.subplots(figsize=(5,5))
sns.set(style = "whitegrid")
sns.violinplot(x = "Hardness", y=dfalloys, data=df, ax=axes, orient='h')



def set_axis_style(ax, labels):
    ax.get_yaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks(np.arange(0, 800))
    ax.set_yticklabels(labels)
    ax.set_ylim(0.25, len(labels) + 0.75)
    ax.set_ylabel('Sample name')

set_axis_style(ax,['Al','Co','Cu','Cr','Fe', 'Ni'][::-1])
plt.show()

'''
