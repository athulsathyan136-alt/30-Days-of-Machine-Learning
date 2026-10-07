import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/customers.csv')

print('\n======DATASET========')
print(df)

X = df[['age','monthly_spend','months_with_company','support_calls']]

y = df['churn']

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print('\n=======DATA SPLIT==========')
print('Training sample ',len(X_train))
print('Testing sample ',len(X_test))

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42


)

model.fit(X_train,y_train)

print('\n=====MODEL TRAINED========')

prediction = model.predict(X_test)

print('\n========TEST PREDICTIONS===========')

for actual, predicted in zip(y_test,prediction):
    print(f'Actual :{actual} | predicted :{predicted}')

accuracy = accuracy_score(y_test,prediction)

print('\n========MODEL EVALUTION=========')
print('Accuracy: ',round(accuracy *100 ,2))

print('\n=========FEATURE IMPORTANCE=========')

for feature,importance in zip(X.columns,model.feature_importances_):
    print(feature ,':', round(importance,4))

new_customer = pd.DataFrame({
    "age": [29],
    "monthly_spend": [40],
    "months_with_company": [6],
    "support_calls": [4]
})

new_prediction = model.predict(new_customer)

new_pro = model.predict_proba(new_customer)

print('\n=======NEW CUSTOMERS PREDICTION==========')
print('Age :',new_customer['age'].iloc[0])
print('Monthly spend :',new_customer['monthly_spend'].iloc[0])
print('month with company :',new_customer['months_with_company'].iloc[0])
print('Support Calls :',new_customer['support_calls'].iloc[0])
print('Predicted class :',new_prediction[0])
print('Probability of staying :',round(new_pro[0][0]*100 ,2),'%')
print('Probability of leaving :',round(new_pro[0][0]*100 ,2),'%')

if new_prediction[0] == 0:
    print('Customer will likely STAY')
else:
    print('Customers will likely LEAVE')