import pandas as pd
import numpy as np

df = pd.read_csv('data/health_data.csv')

print('======DATASET=========')
print(df)

X = df[['age',
       'bmi',
       'glucose',
       'blood_pressure']].values

y = df['risk'].values

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean)/X_std

X_scaled = np.column_stack((np.ones(len(X_scaled)),X_scaled)
)
np.random.seed(42)

indices = np.random.permutation(len(X_scaled))

split = int(len(X_scaled)*0.8)

train_indices = indices[:split]
test_indices = indices[split:]

X_train = X_scaled[train_indices]
y_train = y[train_indices]

X_test = X_scaled[test_indices]
y_test = y[test_indices]

print('\n=====DATA SPLIT======')
print('Training sample: ',len(X_train))
print('Testing sample: ',len(X_test))

weight = np.zeros(X_train.shape[1])

learning_rate = 0.1
epochs = 2000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for epoch in range(epochs):
    linear_output = X_train @ weight
    probabilies = sigmoid(linear_output)

    error = probabilies - y_train


    gradient = (X_train.T @ error) / len(X_train)

    weight = weight - learning_rate * gradient

print('\n=======MODEL TRAINED==========')
print('Weight: ',weight)

test_prob = sigmoid(X_test @ weight)
test_pre  = (test_prob >= 0.5).astype(int)

print('\n========TEST PREDICTIONS==========')
for actual,probability,predicted in zip(y_test,test_prob,test_pre):
    print(f"Actual: {actual} | Probability: {probability} | Predicted: {predicted}")

accuracy = np.mean(test_pre == y_test)

print('\n====== MODEL EVALUTION=============')
print('Accuracy :',round(accuracy * 100,2),"%")

new = np.array([[39,28.5,130,135]])

new_per_scaled = (new - X_mean)/X_std

new_per_scaled = np.column_stack((np.ones(len(new_per_scaled)),new_per_scaled))

new_pro = sigmoid(new_per_scaled @ weight)[0]

new_pre = int(new_pro >= 0.5)

print('\n=====NEW PERSON PREDICTION=======')
print('Age: ',new[0][0])
print('BMI: ',new[0][1])
print('Glucose: ',new[0][2])
print('Blood Pressure: ',new[0][3])

print('Probability: ',round(new_pro,2))

print('Predicted; ',new_pre)

if new_pre == 0:
    print('Predicted class : LOW RISK')
else:
    print('Predicted class : HIGH RISK')

