#OLS METHOD
#IMPORT LIBRARIES
import numpy as nb #to handel numeric part
import pandas as pd #to read file 
import matplotlib.pyplot as plt #for visualization
import seaborn as sns  #for visualization
#Importing the Dataset
dataset = pd.read_excel("slr10.xls")
X = dataset.iloc[ : , 0]
Y = dataset.iloc[ : , 1]
#Plooting scatter data
plt.scatter(X, Y)
plt.show()
#Training the Model
X_mean = nb.mean(X)
Y_mean = nb.mean(Y)
n = 0
d = 0
for i in range(len(X)):
  n = n + (X[i] - X_mean)*(Y[i] - Y_mean)
  d = d + (X[i] - X_mean)**2
m = n / d
c = Y_mean - m*X_mean

print (m, c)
#Predicting the dependent variable
Y_pred = m*X + c

plt.scatter(X, Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')
plt.show()
