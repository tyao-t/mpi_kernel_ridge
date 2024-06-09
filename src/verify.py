import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel

file_path = 'data/housing.tsv'
data = pd.read_csv(file_path, sep='\t', header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=17)
# print(x_train[:3])
x_train = scaler.fit_transform(x_train)
y_train = y_train.to_numpy().reshape((-1, 1))
y_train = scaler.fit_transform(y_train)
# for i in range(len(y)): y[i] = [y[i]]
# print(y.shape)
# y_train = scaler.transform(y_train)

# print(x_scaled[:2, :], y_scaled[:2])

alpha = 1.0
sigma = 1.0  
gamma = 1 / (2 * sigma**2)  # 计算 gamma

# K = rbf_kernel(x_train, x_train, gamma=gamma)

# # print("K:\n", K)

# n_samples = x_train.shape[0]
# K_reg = K + alpha * np.eye(n_samples)

# # print("K_reg:\n", K_reg)

# dual_coef_ = np.linalg.inv(K_reg).dot(y_train)

# print("dual_coef_ :\n", dual_coef_)

from sklearn.kernel_ridge import KernelRidge
model = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
model.fit(x_train, y_train)
print(model.dual_coef_.shape)
print("KernelRidge 的 dual_coef_:\n", model.dual_coef_[:10])

exit()

X_avg = X.mean()
y_avg = y.mean()
X_var = X.var(ddof=1)
y_var = y.var(ddof=1)

print(X_avg)
print(y_avg)

print(X_var)
print(y_var)