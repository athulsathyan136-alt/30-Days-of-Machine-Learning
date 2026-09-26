import pandas as pd
import numpy as np

df = pd.read_csv('data/house_prices.csv')

print('=====DATASET=========')
print(df)

X = df[['area_sqft','bedrooms']].values

y =df['price_lakh'].values

np.random.seed(42)
indices = np.random.permutation(len(df))

split_index = int(len(df)*0.8)

train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train = X[train_indices]
y_train = y[train_indices]

X_test = X[test_indices]
y_test = y[test_indices]

print("\n====== TRAINIG DATA ==========")
print('Training samples: ',len(X_train))
print(X_train)
print(y_train)

print("\n====== TEST DATA ==========")
print('Training samples: ',len(X_test))
print(X_test)
print(y_test)

train_area = X_train[:,0]

area_mean = train_area.mean()
price_mean = y_train.mean()

numerator = ((train_area - area_mean) * (y_train - price_mean)).sum()
denominator = ((train_area - area_mean) ** 2).sum()

slope = numerator/denominator
intercept = price_mean - (slope*area_mean)

print('=======MODEL=========')
print('Slope: ',slope)
print('Intercept: ',intercept)

test_area = X_test[:,0]
predictions = slope  * test_area + intercept

print('=======TEST PREDICTIONS=========')

for actual,predicted in zip(y_test,predictions):
    print(f"Actual: {actual:.2f} lakh | Predicted : {predicted:.2f} lakh")

mae = np.mean(np.abs(y_test - predictions))

print('\n====TEST RESULT========')
print('Mean Absolute Error: ',round(mae,2),"lakh")

new_area = 1550
new_predict = slope*new_area + intercept

print('\n======NEW HOUSE======')
print('Area: ',new_area,"sqft")
print('Predicted price: ',round(new_predict,2),"lakh")