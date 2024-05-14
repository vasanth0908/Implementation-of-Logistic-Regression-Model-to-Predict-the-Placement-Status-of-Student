# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results.
## Program:

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.       
Developed by:vasanth   
RegisterNumber: 212222110052  
*/

```
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
## Opening File:

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/56a0c7c3-65ac-4153-9597-1780eff84f9e)

## Droping File:

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/3a4bebe7-76de-4603-b306-748fd0abd048)

## Duplicated():

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/ad02d60b-ea15-402c-bda2-700351b67483)

## Label Encoding:

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/ff072a69-3f02-4558-975b-17452d4a1948)

## Spliting x,y:

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/7d66b007-2f9f-4302-b12c-235fa42c1cf7)

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/87b8d534-0a91-4a40-9894-7afd258174ec)

## Prediction Score

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/4741b9eb-9815-4eb4-9195-2531313df3eb)

## Testing accuracy

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/f5a5f78b-f3ed-4bc4-bfc3-b50db834fee0)

## Classification Report

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/c5a73c16-f062-4226-aca2-dc3bf14f67c1)

## Testing Model:

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/8a085f1c-3a52-4812-a4b8-d8ff199f2534)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
