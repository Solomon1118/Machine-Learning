import pandas as pd
import numpy as np
df = pd.read_csv("/content/Dataset_03.csv")

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

"""#Splitting Dataset"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.2, random_state = 1)

"""#Perform Feature Scaling"""

#Importing StandardScaler
from sklearn.preprocessing import StandardScaler
#Creating Instance of StandarScaler
sc = StandardScaler()
#Perform scaling in X_train with fit_transform.
X_train.iloc[:, 4:] = sc.fit_transform(X_train.iloc[:, 4:])

#Perform scaling in X_test with fit_transform.
X_test.iloc[:, 4:] = sc.transform(X_test.iloc[:, 4:])

dataset = sc.fit_transform(df.iloc[:, 4:])

"""#Dataset Before Scaling"""

df.head()

"""#Dataset After Scaling"""

dataset[:5]