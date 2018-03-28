import numpy as np
import pandas as pd
import statsmodels.api as sm


data = pd.read_csv('1.csv')
dataset1 = np.array(data)
X1 = dataset1[:, 0:5]
Y1 = dataset1[:, 6]
X1 = sm.add_constant(X1)
est = sm.OLS(Y1, X1).fit()
print(est.summary())

dataset = np.array(data)
cor = np.corrcoef(dataset, rowvar=0)[:, 0]
print(cor)

