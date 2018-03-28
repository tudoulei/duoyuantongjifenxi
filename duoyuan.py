# 导入类库
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

'''
from numpy import arange
import scipy as sp
import statsmodels.api as sm
import scipy.stats as stats
import scipy.optimize as opt
from pandas import  set_option
from pandas import  set_option
from pandas.plotting import scatter_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
'''

# 导入数据
the_amount_of_data = 10000
filename = '1.csv'
names = ['band1', 'band2', 'band3', 'band4', 'band5', 'band6', 'band7']
data = read_csv(filename, names=names, delimiter=',', nrows=the_amount_of_data)


# 线性回归 显示一元线性回归模型
def yiyuan(rows):
    filename = '1.csv'
    names = ['band1', 'band2', 'band3', 'band4', 'band5', 'band6', 'band7']
    data = read_csv(filename, names=names, delimiter=',', nrows=rows)
    sns.pairplot(data, x_vars=['band1', 'band2', 'band3', 'band4', 'band5', 'band6', ], y_vars='band7', size=7,
                 aspect=0.8, kind='reg')
    plt.show()

# Scatter plot of only the highly correlated pairs
# 多分类的数据更有效
def scatter(rows):
    filename = '1.csv'
    names = ['band1', 'band2', 'band3', 'band4', 'band5', 'band6', 'band7']
    data = read_csv(filename, names=names, delimiter=',', nrows=rows)
    # Scatter plot of only the highly correlated pairs
    sns.pairplot(data, hue="band3", size=6, x_vars='band1', y_vars='band2')
    plt.show()
# 多元线性回归
def duoyuan(X, Y):
    validation_size = 0.2
    seed = 7
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
    linreg = LinearRegression()
    model = linreg.fit(X_train, Y_train)
    print(model)
    print('常系数：%.9f' % linreg.intercept_)
    print('系数为：')
    print(linreg.coef_)
    predictions = model.predict(X_validation)
    # 求残差
    error = []
    for i, prediction in enumerate(predictions):
        error.append(Y_validation[i] - prediction)
    plt.plot(error)
    plt.show()

    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值

    print(" 残差最小值  : %f\n 残差一分位数:%f\n 残差中位数  :%f\n 残差三分位数:%f\n 残差最大值  :%f" % (
        np.min(error), np.percentile(error, 25), np.median(error), np.percentile(error, 75), np.max(error)))
    print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE
    print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))  # 均方根误差RMSE
    print("MAE = ", sum(absError) / len(absError))  # 平均绝对误差MAE
    R_Square = model.score(X_validation, Y_validation)
    n = 10000
    p = 7
    plt.boxplot(error)
    plt.show()
    Adjust_R_Square = 1 - (1 - R_Square) * (n - 1) / (n - p - 1)
    print('R-squared: %.2f' % R_Square)
    print('R-squared: %.2f' % Adjust_R_Square)
    t, p = ttest_ind(Y_validation, predictions)
    print('t检验为: %f' % t)
    print('p检验为: %f' % p)
    print('****************************************************************')



# 多元线性回归---band3-band1 band2
array = data.values
X = array[:, 0:5]
Y = array[:, 6]
print('多元线性回归---band7-band1 band2......band5')
duoyuan(X, Y)


'''
# 多元线性回归---band1 band1- band2 band3
array = data.values
X = array[:, 1:3]
Y = array[:, 0]
print('多元线性回归---band1- band2 band3')
duoyuan(X, Y)

# 多元线性回归---band1作因变量，其他作自变量 band1-band2 band3 band4 band5 band6 band7
array = data.values
X = array[:, 0:6]
Y = array[:, 6]
print('band1-band2 band3 band4 band5 band6 band7')
duoyuan(X, Y)

#  多元线性回归---band1作因变量，其他作自变量 band1-band3 band6 band7
array = data.values
X = array[:, (2,5,6)]
Y = array[:, 0]
print('band1-band3 band6 band7')
duoyuan(X, Y)

# 多元线性回归---band1作因变量，其他作自变量 band1-band3 band4 band5 band7
array = data.values
X = array[:, (2,3,4,6)]
Y = array[:, 0]
print('band1-band3 band4 band5 band7')
duoyuan(X, Y)

# 多元线性回归---band1作因变量，其他作自变量 band7-band1 band2 band3
array = data.values
X = array[:, 0:3]
Y = array[:, 6]
print('band7-band1 band2 band3')
duoyuan(X, Y)

'''

'''
X = array[:, (0,1)]
Y = array[:, 2]
est=sm.OLS(Y, X).fit()
est.summary()
'''

'''
# 回归问题的评价测度
plt.plot(Y_validation, 'b', label="validation")
plt.plot(predictions, 'r', label="prediction")
plt.show()
plt.plot(range(len(prediction),Y_validation,'r',label="test")
plt.legend(loc="upper right") #显示图中的标签
plt.xlabel("the number of sales")
plt.ylabel('value of sales')
plt.show()
'''

''' 
# 评估算法——箱线图
y_pred = linreg.predict(X_validation)
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(y_pred)
plt.show()  
'''
