import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#loading the dataset
from sklearn.datasets import load_wine
dataset = load_wine()

df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
df.head()

df.shape

#x is the feature and y is the target
x=df
y=dataset.target

print(x)
print(y)

"""#Splitting Data"""

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
print(x.shape,x_train.shape,x_test.shape)

"""#Standardization"""

print(dataset.data.std())
scaler=StandardScaler()

scaler.fit(x_train)

x_train_standardized=scaler.transform(x_train)
print(x_train_standardized)

x_test_standardized=scaler.transform(x_test)
print(x_train_standardized.std())

print(x_test_standardized)

print(x_test_standardized.std())
