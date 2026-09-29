import pandas as pd
import numpy as np
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)

df = pd.read_csv('data/regression_data.csv')

print('=======DATASET=========')
print(df)

actual = df["actual"]
predicted = df["predicted"]

mae = mean_absolute_error(actual,predicted)

mse = mean_squared_error(actual,predicted)

rmse = np.sqrt(mse)

r2 = r2_score(actual,predicted)

print('\n=======REGRESSION EVALUTION=========')
print(f'MAE: {round(mae,2)}')
print(f'MSE: {round(mse,2)}')
print(f'RMSE: {round(rmse,2)}')
print(f'R2: {round(r2,2)}')
