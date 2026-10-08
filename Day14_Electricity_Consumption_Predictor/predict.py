import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score

df = pd.read_csv('data/energy_data.csv')
print('\n=========DATASET===========')
print(df)

X = df[['temperature','humidity','households','solar_generation']]

y = df['consumption_kwh']

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print('\n=========DATA SPLIT========')
print('Training sample: ',len(X_train))
print('Testing sample: ',len(X_test))

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train,y_train)

print('\n=======MODEL TRAINED========')

prediction = model.predict(X_test)

print("\n====== TEST PREDICTIONS ======")

for actual , predicted in zip(y_test,prediction):
    print(f'Actual :{actual} kWh| Predicted :{predicted} kWh')


mae = mean_absolute_error(y_test,prediction)
r2 = r2_score(y_test,prediction)

print("\n====== MODEL EVALUATION ======")
print('MAE: ',round(mae,2),'%')
print('R2 score:',round(r2,2),'%')

print("\n====== FEATURE IMPORTANCE ======")
for feature,important in zip(X.columns,model.feature_importances_):
    print(feature, ':', round(important,2))

new_data = pd.DataFrame({
    "temperature": [30],
    "humidity": [52],
    "households": [150],
    "solar_generation": [285]
})

new_predict = model.predict(new_data)

print("\n====== NEW PREDICTION ======")
print('Tempature: ',new_data['temperature'].iloc[0])
print('Humidity: ',new_data['humidity'].iloc[0])
print('Households: ',new_data['households'].iloc[0])
print('Solar generation: ',new_data['solar_generation'].iloc[0])
print('Predicted consumption: ', round(new_predict[0],2),'%')