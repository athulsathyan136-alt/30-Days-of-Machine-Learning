import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score

df = pd.read_csv('data/salary_data.csv')

print('======DATASET=======')
print(df)

X = df[["years_experience"]]
y = df["salary_lakh"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print('\n======DATA SPLIT========')
print('Training sample: ',len(X_train))
print('Testing sample: ',len(X_test))

model = LinearRegression()

model.fit(X_train,y_train)

predictions = model.predict(X_test)

print('\n========TEST PREDICTIONS========')
for actual , predicted in zip(y_test,predictions):
    print(f"Actual: {actual:.2f} lakh | Predicted {predicted:.2f} lakh")

mae = mean_absolute_error(y_test,predictions)
r2 = r2_score(y_test,predictions)

print('\n=====MODEL EVALUTION========')
print("MAE: ",round(mae,2), "lakh")
print("R2 : ",round(r2,4))

new_employee = pd.DataFrame({"years_experience":[7.5]
})

new_predict = model.predict(new_employee)

print('\n ======NEW SALARY PREDICTION==========')
print('Year of experience :',new_employee["years_experience"].iloc[0])
print('Predicted salary: ',round(new_predict[0],2), 'lakh')

