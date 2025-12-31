import pandas as pa
import numpy as np

df = pa.read_csv('/content/apples_and_oranges.csv')

df.head()

print(df.dtypes)

df.describe(include = "all")

print("Median of 'Weight':", df['Weight'].median())
print("Median of 'Size':", df['Size'].median())

df.info()
