# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: VIKASH S
RegisterNumber: 212222240115 
*/
```
``` python

import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()
data.info()
data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
l0=LabelEncoder()

data["Position"]=l0.fit_transform(data['Position'])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])


```

## Output:

## INITIAL DATASET
![image](https://github.com/vikashsenthil21/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119433834/18659024-45b8-41ed-acb0-7051531f5f7f)

## DATA.INFO ()
![image](https://github.com/vikashsenthil21/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119433834/9a802fd3-9610-4209-b2ac-9cfc30f969a0)


## OPTIMIZATION OF NULL VALUES
![image](https://github.com/vikashsenthil21/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119433834/35d0f8c3-ba05-4155-980f-c398a2fc6da7)

## Converting string literals to numerical values using label encoder:
![image](https://github.com/vikashsenthil21/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119433834/d7b8be15-63d0-41c8-8ae6-247aacde85c2)

## MSE Value
![image](https://github.com/vikashsenthil21/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119433834/94a2e5b7-a052-4480-8200-614b3fc9711a)

## R2 VALUE (VARIANCE) :
![image](https://github.com/vikashsenthil21/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119433834/44c02ea8-bbbf-4754-93b1-b120bb280e99)


## DATA PREDICTION
![image](https://github.com/vikashsenthil21/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119433834/a1b3dcea-2c13-4300-8fee-17c1a834f406)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
