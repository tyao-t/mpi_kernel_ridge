import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

train_data = pd.read_csv('data/housing_train.tsv', sep='\t', header=None)
test_data = pd.read_csv('data/housing_test.tsv', sep='\t', header=None)

x_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
x_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train_scaled = scaler_x.fit_transform(x_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

x_test_scaled = scaler_x.transform(x_test)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

model = KernelRidge(alpha=1.0, gamma=1.0, kernel='rbf')
model.fit(x_train_scaled, y_train_scaled.ravel())

y_pred_scaled = model.predict(x_test_scaled)

y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

y_test_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))  
rmse = np.sqrt(mean_squared_error(y_test_true, y_pred))
print(f'RMSE: {rmse}')