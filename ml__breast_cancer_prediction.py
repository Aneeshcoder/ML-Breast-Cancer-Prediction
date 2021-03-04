import warnings
warnings.filterwarnings('ignore') #ignore the warnings due to time lack or timeline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #use to plot functions as graphs

df = pd.read_csv('https://raw.githubusercontent.com/Aneeshcoder/ML-Breast-Cancer-Prediction/main/data.csv') #reading raw data

df.head() #checking data with their heads or titles & df is data frame

df.columns #name of the columns

df.info() #checking data types

#drop column axis=1 and row axis=0
#we need to drop unnamed column
df['Unnamed: 32']

df = df.drop("Unnamed: 32", axis=1)

df.head()

df.columns

#id column not providing any useful data
df.drop("id",axis=1, inplace=True) 
# df=df.drop('id',axis=1) ,we can also use this

df.columns

type(df.columns) #data type of df

l = list(df.columns) #convert columns into list
print(l)

#now we can use index location that is 0,1,2,....
features_mean = l[1:11] #in python start index is inclusive and end index is exclusive

features_se = l[11:21]

features_worst = l[21:] #feature is known as column in ML

print(features_mean)

print(features_se)

print(features_worst)

df.head(2)

df['diagnosis'].unique()
# M= Malignant cancer and B= Benign cancer

sns.countplot(df['diagnosis'], label="count",);

df['diagnosis'].value_counts()
#value count function tells us about values of M and B in data set

df.shape #569 no. of rows and 31 no. of columns

#Explore the Data

df.describe()
#summary of all the numeric columns
#std is standard deviation

#Create Correlation Plot  ,, Correlation values[-1,1]
# +ve correlation = if 1st variable increase then 2nd variable also increases
# -ve correlation = if 1st variable increase then 2nd variable decreases
# no correlation = zero or nothing

len(df.columns)

#correlation plot 
corr = df.corr()
corr

corr.shape

plt.figure(figsize=(8,8)) #(8,8) is the size of graph i.e. 8X8
sns.heatmap(corr);

df.head()

#M==1 and B==0
df['diagnosis']=df['diagnosis'].map({'M':1, 'B':0})

df.head()

df['diagnosis'].unique()

X = df.drop('diagnosis',axis=1)  #Capital X is preferred
X.head()

y = df['diagnosis']  #Small y is preferred
y.head()

#0. fit()
#1. Study(X_train,y_train)
#2. Brain(intelligence)
#3. How good is preparation X_test
#4. y_pred i.e. predicted values
#5. Comparison of y_pred and y_test => how good is model prepared

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

df.shape

X_train.shape

X_test.shape

y_train.shape

y_test.shape

from sklearn.preprocessing import StandardScaler
ss = StandardScaler() #scaling near to 0
X_train = ss.fit_transform(X_train)  #fit and immediately transform
X_test = ss.transform(X_test) #dont need fit as fit is equivalent to studying

X_train

# Machine Learning Models

# Logistic Regression

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred =lr.predict(X_test)

y_pred

y_test

# accuracy_score gives performance of function

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))  # approx 98.245%

lr_acc = accuracy_score(y_test, y_pred)
print(lr_acc)

results = pd.DataFrame()
results  #empty dataframe

tempResults = pd.DataFrame({'Algorithm':['Logistic Regression Method'], 'Accuracy':[lr_acc]}) #Headings Algorithm and Accuracy
results = pd.concat([results,tempResults])
results = results[['Algorithm','Accuracy']]
results

# Decision Tree Classifier //more powerful algorithm than LR

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

dtc_acc = accuracy_score(y_test, y_pred)
print(dtc_acc)

tempResults = pd.DataFrame({'Algorithm':['Decision Tree Classifier Method'], 'Accuracy':[dtc_acc]}) #Headings Algorithm and Accuracy
results = pd.concat([results,tempResults])
results = results[['Algorithm','Accuracy']]
results

# Random Forest Classifier //most powerful algorithm

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

rfc_acc = accuracy_score(y_test, y_pred)
print(rfc_acc)

tempResults = pd.DataFrame({'Algorithm':['Random Forest Classifier Method'], 'Accuracy':[rfc_acc]}) #Headings Algorithm and Accuracy
results = pd.concat([results,tempResults])
results = results[['Algorithm','Accuracy']]
results

# Support Vector Classifier

from sklearn import svm
svc = svm.SVC()
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

svm_acc = accuracy_score(y_test, y_pred)
print(svm_acc)

tempResults = pd.DataFrame({'Algorithm':['Support Vector Classifier Method'], 'Accuracy':[svm_acc]}) #Headings Algorithm and Accuracy
results = pd.concat([results,tempResults])
results = results[['Algorithm','Accuracy']]
results
