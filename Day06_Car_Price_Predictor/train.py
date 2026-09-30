import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error

df = pd.read_csv('data/car_prices.csv')

print('=========DATASET==========')
print(df)

X = df[[
    'car_age',
    'mileage_km',
    'engine_cc',
    'owner_count'
]]

y = df["price_lakh"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print('Train dataset lenth: ',len(X_train))
print('Test dataset length: ',len(X_test))

model = LinearRegression()
model.fit(X_train,y_train)

prediction = model.predict(X_test)

print('\n=======TEST PREDICTION==========')
for actual,predicted in zip(y_test,prediction):
    print(f"Actual :{actual} lakh | Predicted : {predicted}")
    
mae = mean_absolute_error(y_test,prediction)
r2 = r2_score(y_test,prediction)

print('\n=====MODEL PREDICTON=========')
print('MAE : ',round(mae,2),'lakh')
print('R2 ',round(r2,2))

new_car = pd.DataFrame({
    "car_age": [3],
    "mileage_km": [45000],
    "engine_cc": [1800],
    "owner_count": [1]
})

new_pre = model.predict(new_car)


print("\n====== NEW CAR PREDICTION ======")
print("Car age ",new_car["car_age"].iloc[0],"years")
print("Mileage ",new_car["mileage_km"].iloc[0],"km")
print("Engine ",new_car["engine_cc"].iloc[0],"cc")
print("Owners ",new_car["owner_count"].iloc[0])
print("Predicted price",round(new_pre[0],2),"lakh")