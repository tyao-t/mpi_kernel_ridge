import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

train_data = pd.read_csv('data/housing_train.tsv', sep='\t', header=None)
test_data = pd.read_csv('data/housing_test.tsv', sep='\t', header=None)

def encode_ocean_proximity(x):
    # Extract the last column and reshape it to a 2D array
    ocean_proximity_column = x[:, -1].reshape(-1, 1)
    
    # Create an instance of the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # Set sparse=False to get a dense array
    
    encoded_column = encoder.fit_transform(ocean_proximity_column)
    x_without_ocean = x[:, :-1]
    x_encoded = np.hstack((x_without_ocean, encoded_column))
    
    return x_encoded

def process_ocean(x, y, fit=True):
    global ocean_avg
    if fit:
        ocean_avg = {}
        for i in range(5): ocean_avg[i] = [0, 0]
        for i in range(x.shape[0]):
            ocean_avg[int(x[i][8])][0] += y[i][0]
            ocean_avg[int(x[i][8])][1] += 1
        for i in range(5): ocean_avg[i] = ocean_avg[i][0]/ocean_avg[i][1]

    print(ocean_avg)
    # ocean_avg = {0: 1, 1: -1, 2: 1.1, 3: 1.2, 4: 1.5}
    for i in range(x.shape[0]): x[i][8] = ocean_avg[int(x[i][8])]
        
def add_housing_pressure(x):
    for i in range(x.shape[0]):
        x[i][3] /= x[i][5]
        x[i][4] /= x[i][5]
        x[i][6] = x[i][5]/x[i][6]

def scale_income(x): 
    for i in range(x.shape[0]): 
        x[i][0] *= 27.5
        x[i][1] *= 27.5
        x[i][5] *= 0.125
        x[i][6] *= 1.5
        x[i][8] *= 1.5
        x[i][3] *= 1
        x[i][4] *= 1

x_train = train_data.iloc[:, :-1].values
# x_train = encode_ocean_proximity(x_train)
add_housing_pressure(x_train)
# x_train = np.delete(x_train, 4, axis=1)
y_train = train_data.iloc[:, -1].values.reshape(-1, 1)
process_ocean(x_train, y_train, fit=True)
x_test = test_data.iloc[:, :-1].values
# x_test = encode_ocean_proximity(x_test)
add_housing_pressure(x_test)
# x_test = np.delete(x_test, 4, axis=1)
# print(x_train.shape)
y_test = test_data.iloc[:, -1].values.reshape(-1, 1)
process_ocean(x_test, y_train, fit=False)
# 标准化特征

x_scaler = StandardScaler()
# print(x_train[:3, :])
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.transform(x_test)
# print(x_train_scaled[:3, :])
scale_income(x_train_scaled)
scale_income(x_test_scaled)

# 标准化目标值
y_scaler = StandardScaler()
y_scaler2 = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
pt = PowerTransformer(method='box-cox', standardize=True)
# y_train_box_cox = pt.fit_transform(y_train)
# print(y_train_box_cox.shape)
# print(y_train_box_cox[:5])
# exit()
# print(y_train[:10], y_train_scaled[:10], y_train_box_cox[:10], np.log(y_train)[:10])
# 定义Kernel Ridge Regression模型

params = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

# rf_model = RandomForestRegressor(n_estimators=200, max_features='sqrt', random_state=199)
# rf_model.fit(x_train_scaled, y_train_box_cox.ravel())
# feature_importances = rf_model.feature_importances_
# print(feature_importances)

# param_grid = {
#     'alpha': np.geomspace(10**-5, 10**3, 25),
#     'gamma': np.geomspace(10**-3, 10**3, 19)
# }

search_space = {
    'alpha': Real(1e-4, 1e2, prior='log-uniform'),
    'gamma': Real(1e-2, 1e2, prior='log-uniform')
}

pbounds = {
    'alpha': (0.0085, 0.315),
    "gamma": (0.0085, 0.315)
}

# pbounds = {
#     'alpha': (0.0085, 0.315),
#     "gamma": (0.0085, 0.315)
# }

def krr_cv(alpha, gamma):
    # 创建Kernel Ridge Regression模型
    model = KernelRidge(alpha=alpha, gamma=gamma, kernel='rbf')
    
    # 拟合模型
    model.fit(x_train_scaled, y_train_scaled)
    
    # 预测
    y_pred = model.predict(x_test_scaled)
    # print(x_train_scaled[:2, :])
    # print(y_train_box_cox[:2])
    # print(y_pred[:2])
    y_pred_scaled_back = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
    # print(y_pred_scaled_back[:5])
    # print(y_test[:5])
    # 计算均方误差
    rmse = root_mean_squared_error(y_test, y_pred_scaled_back)
    # 返回负均方误差
    return -rmse
# model = KernelRidge(alpha=0.1, kernel='rbf', gamma=0.05)

# model2 = KernelRidge(kernel='rbf')
# model3 = KernelRidge(kernel='rbf')

# 训练模型
# model.fit(x_train_scaled, y_train.ravel())
# bayes_search2 = BayesSearchCV(
#     estimator=model2,
#     search_spaces=search_space,
#     n_iter=8,
#     random_state=42,
#     cv=5,
#     n_jobs=-1,
#     verbose=2
# )
# bayes_search2.fit(x_train_scaled, y_train_scaled)
# bayes_search3 = BayesSearchCV(
#     estimator=model3,
#     search_spaces=search_space,
#     n_iter=96,
#     random_state=42,
#     cv=5,
#     n_jobs=5,
#     verbose=2
# )

optimizer = BayesianOptimization(
    f=krr_cv,
    pbounds=pbounds,
    random_state=46,
)

optimizer.maximize(
    init_points=16,  # 初始的随机探索次数
    n_iter=96,      # 优化循环的迭代次数
)

print("最佳参数：", optimizer.max['params'])
print("对应的目标函数值：", optimizer.max['target'])
model = KernelRidge(alpha=0.017765911413670298, gamma=0.13659263195982618, kernel="rbf")
# model = KernelRidge(alpha=0.0405194803069227, gamma=0.1763712749312077, kernel="rbf")
model.fit(x_train_scaled, y_train_scaled)
y_train_pred = model.predict(x_train_scaled)
y_pred_test_boxcox = model.predict(x_test_scaled)
# bayes_search3.fit(x_train_scaled, y_train_box_cox)

# 进行预测
# y_pred_test = model.predict(x_test_scaled)
# y_pred_test_scaled = bayes_search2.best_estimator_.predict(x_test_scaled)
# y_pred_test_boxcox = bayes_search3.best_estimator_.predict(x_test_scaled)

# 逆标准化预测值
# y_pred_test_scaled_back = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1))
y_train_back = pt.inverse_transform(y_train_pred.reshape(-1, 1))
y_pred_test_boxcox_back = pt.inverse_transform(y_pred_test_boxcox.reshape(-1, 1))

# 计算RMSE
# test_rmse= root_mean_squared_error(y_test, y_pred_test)

# print(bayes_search2.best_params_)
# print(bayes_search3.best_params_)
# test_rmse_scaled = root_mean_squared_error(y_test, y_pred_test_scaled_back)
train_rmse_boxcox = root_mean_squared_error(y_train, y_train_back)
test_rmse_boxcox = root_mean_squared_error(y_test, y_pred_test_boxcox_back)

# 输出结果
# print(y_test[:10])
# print(y_pred_test[:10], y_pred_test_scaled_back[:10].ravel())
# print("Test RMSE (unscaled):", test_rmse)
# print("Test RMSE (scaled):", test_rmse_scaled)
print("Train RMSE (box-cox):", train_rmse_boxcox)
print("Test RMSE (box-cox):", test_rmse_boxcox)
