# 导入类库
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt

from pandas import read_csv
from pandas import  set_option
from pandas.plotting import scatter_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
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
# 导入数据
filename = '1.csv'
names = ['band1', 'band2', 'band3', 'band4', 'band5', 'band6', 'band7']
data = read_csv(filename, names=names, delimiter=',', nrows=10000)

'''
# 数据维度
print(data.shape)
# 特征属性的字段类型
print(data.dtypes)
# 查看最开始的30条记录
set_option('display.line_width', 120)
print(data.head(30))
# 描述性统计信息
set_option('precision', 1)
print(data.describe())
# 关联关系
set_option('precision', 2)
print(data.corr(method='pearson'))
# 直方图
data.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
pyplot.show()
# 密度图
data.plot(kind='density', subplots=True, layout=(4,4), sharex=False, fontsize=1)
pyplot.show()
#箱线图
data.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, fontsize=8)
pyplot.show()
# 散点矩阵图
scatter_matrix(data)
pyplot.show()


# 相关矩阵图
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(data.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = np.arange(0, 7, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()


# 分离数据集
array = data.values
X = array[:, 0:6]
Y = array[:, 6]

# 单变量特征选定
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
print(fit.scores_)
features = fit.transform(X)
print(features)

# 递归特征消除
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print('特征个数：')
print(fit.n_features_)
print('被选定的特征：')
print(fit.support_)
print('特征排名：')
print(fit.ranking_)
'''

print(data.skew())
#data.groupby('Cover_Type').size()

array = data.values
X = array[:, 0:6]
Y = array[:, 6]
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,test_size=validation_size, random_state=seed)

'''
# Scatter plot of only the highly correlated pairs
for band1,band2,band3 in data:
    sns.pairplot(data, hue="Cover_Type", size=6, x_vars=cols[i],y_vars=cols[j] )
    plt.show()
'''








'''num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'
models = {}
models['LR'] = LinearRegression()
models['LASSO'] = Lasso()
models['EN'] = ElasticNet()
models['KNN']  = KNeighborsRegressor()
models['CART'] = DecisionTreeRegressor()
models['SVM'] = SVR()
# 评估算法
results = []
for key in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_result = cross_val_score(models[key], X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_result)
    print('%s: %f (%f)' % (key, cv_result.mean(), cv_result.std()))
#评估算法——箱线图
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(models.keys())
pyplot.show()'''
