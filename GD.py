#GRADIENT DESCENT METHOD
#IMPORT LIBRARIES
import numpy as nb #to handel numeric part
import pandas as pd #to read file 
import matplotlib.pyplot as plt #for visualization
import seaborn as sns  #for visualization
#IMPORTING AND READING FILE
data=pd.read_excel('slr10.xls')
data.head()
#EXTRACTING ORIGINAL VALUES OF X: independent varaiable & Y: dependent variable
 X=data.iloc[:,0]
Y=data.iloc[:,1]
#PANDAS METHOD TO GET VALUES OF CENTRAL TENDENCIES
data.describe()
#PLOTTING OVER ORIGINAL DATASET (SCATTER PLOT)
sns.scatterplot(x='X', y='Y', data=data)
plt.xlabel('Sepal Width')
plt.ylabel('Sepal Length')
plt.title('iris setosa- Scattered Plot')
plt.show()
#DIVIDING DATASET INTO TRAINING: TESTING as 75% n 25%
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test=train_test_split(X, Y, test_size=0.25)
#HYPOTHESIS PARAMETER INTIALIZATION
#(Y=mX+c)
m=0
c=0
L=0.0001  #Learning Rate
epochs=500 #iterartions
n=float(len(x_train))
#TRAINING THE MODEL
e=[]
for i in range (0, epochs): #this loop is for epochs
    for i in range (len(x_train)): #this is to access every value
        y_pred=m*x_train+c
        #print(y_pred)
        e.append((1/n)*(sum((y_pred-y_train)**2))) #mean square error function
        #print(e)
        D_m=(-2/n)*sum(x_train * (y_train-y_pred)) #partial derivative wrt m
        D_c=(-2/n)*sum((y_train-y_pred))  #partial derivative wrt c
        m=m-(L*D_m)  #correcting m value
        c=c-(L*D_c)  #correcting c value
        #print(m,c)
        #print('**********************')
 #FINAL VALUES OF m & c
 print(m,c)
 #PLOT ERROR V/s ITERATION CHANGE
 plt.style.use('seaborn')
plt.title('Reduction in error')
plt.xlabel('Number of epocs')
plt.ylabel('Error')
plt.plot(e)
plt.show()
#PLOTTING PREDICTED VALUE
plt.scatter(x_train,y_pred)
plt.xlabel('Sepal Width')
plt.ylabel('Sepal Length')
plt.title('iris setosa- Prediction')
plt.plot([min(x_train), max(x_train)], [min(y_pred), max(y_pred)], color='red')
plt.show()
#PREDCTION OVER TESTING DATASET
 for i in range (len(x_test)):
    y_t_pred=m*x_test+c
#print(y_t_pred)
e=(1/n)*(sum((y_t_pred-y_test)**2))
print(e*100,"%")    
#SAVING VALUES IN CSV FILE
toSave= pd.DataFrame(data = y_t_pred, columns=['y_t_pred'])
toSave.to_csv('Result_slr10.csv', index =False)
