import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

df = pd.read_csv("TrainOnMe-3.csv") 
dfeval = pd.read_csv("EvaluateOnMe-3.csv") 
print(dfeval.describe())
print(df.describe())

df = df.drop(columns=df.columns[0], axis=1,) # drop id column from trainOnMe
df = df.drop('x11', axis=1) # drop id column from trainOnMe
df = df.drop('x12', axis=1) # drop id column from trainOnMe
df = df.replace({'?':np.nan}).dropna() #drop rows that contain bad data in one or more columns
df['x1'] = df['x1'].astype(float)
df["x13"].replace('0.37.46222', '37.46222', inplace=True)
df['x13'] = df['x13'].astype(float)
new_df = df[
    (np.abs(stats.zscore(df.select_dtypes(include=['float64']))) < 3).all(axis=1)
]
#print(new_df.describe())
out["x6"] = out["x6"].str.lower() #change to lowercase
dfeval["x6"] = dfeval["x6"].str.lower()
out = pd.get_dummies(out, columns=['x6'])
dfeval = pd.get_dummies(dfeval, columns=['x6'])
out = out.drop('x6_syster och brö', axis=1)

X = out[['x1', 'x2', 'x3', 'x4', 'x5', 'x7', 'x8', 'x9', 'x10', 'x13', 'x6_alfvensalen', 'x6_biblioteket', 'x6_entré', 'x6_lindstedvägen 24', 'x6_syster och bro', 'x6_östra station']]  # Training Features
Y = out['y']  # Train Labels
#X_test = dfeval[['x1', 'x2', 'x3', 'x4', 'x5', 'x7', 'x8', 'x9', 'x10', 'x13', 'x6_7-11', 'x6_alfvensalen', 'x6_biblioteket', 'x6_entré', 'x6_lindstedvägen 24', 'x6_syster och bro', 'x6_östra station']]  # Testing Features

'''
for col in X:
    X.boxplot(column=col, by=Y, figsize=(6,6))
    plt.title(col)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(X.describe())
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(new_X.describe())

plt.show()
'''
