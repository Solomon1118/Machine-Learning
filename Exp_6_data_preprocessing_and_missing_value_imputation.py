import pandas as pd
import numpy as np
df = pd.read_csv('/content/record_new.csv')
df.head()

# replace "?" to NaN
df.replace("?", np.nan, inplace = True)
df.head(5)

missing_data = df.isnull()
missing_data.head()

"""#Count Missing Value"""

headers = df.columns
for column in headers:
    print(column)
    print (missing_data[column].value_counts())
    print("")

"""#Column Mean of Age"""

avg_Age = df["Age"].astype('float').mean(axis=0)
print("Average of Age:", avg_Age)

"""#Replace NaN of Age"""

df["Age"].replace(np.nan, avg_Age,inplace = True)
df['Age'].value_counts()

"""#Most frequent value of Age"""

df['Age'].value_counts().idxmax()

df.head()