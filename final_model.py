"""机器学习"""

# 导入类库
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from pandas import read_csv
from scipy.stats import uniform
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from pandas import  set_option
from pandas.plotting import scatter_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
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
the_amount_of_data = 1000
filename = '1.csv'
names = ['band1', 'band2', 'band3', 'band4', 'band5', 'band6', 'band7']
data = read_csv(filename, names=names, delimiter=',', nrows=the_amount_of_data, dtype=float) # 若不加入dtype=float 会出现warning

# 分离自变量与因变量
array = data.values
X = array[:, 0:5]
Y = array[:, 6]

# 分离测试集合+定义常量
validation_size = 0.2
num_folds = 10
seed = 7
scoring = 'r2'
X_train, X_validation, Y_train, Y_validation =  train_test_split(X, Y, test_size=validation_size, random_state=seed)

# 建立模型
models = ElasticNet()
kfold = KFold(n_splits=num_folds, random_state=seed)
results = cross_val_score(models, X_train, Y_train, cv=kfold, scoring=scoring)
print('模型的指标：  %f (%f)' % (results.mean(), results.std()))


'''
# 测试一次模型，使用另外的测试集，此处只选取一部分数据作测试集
Y_test = array[100:200, 6] # 真值
X_test = array[100:200, 0:5]  # 测试x值
model = ElasticNet()
scaler = StandardScaler().fit(X_train)
kfold = KFold(n_splits=num_folds, random_state=seed)
rescaledX_validation = scaler.transform(X_test)
testresult = grid_result.predict(rescaledX_validation) # 不加上这一条，数据是正态化的
# print(model.score(X_test, Y_test))
# print(testresult)
plt.plot(testresult)
plt.plot(Y_test)
plt.show()
'''