import pandas as pd
import numpy as np

df = pd.read_csv('data/customers.csv')

print('======DATASET========')
print(df)

X = df[['age',
        'annual_income',
        'spending_score']].values

y = df['interested'].values

np.random.seed(42)

indices = np.random.permutation(len(X))

split = int(len(X)*0.8)

train_indices = indices[:split]
test_indices = indices[split:]

X_train = X[train_indices]
y_train = y[train_indices]

X_test = X[test_indices]
y_test = y[test_indices]

print('\n====DATA SPLIT======')
print('Training sample ',len(X_train))
print('Testing sample ',len(X_test))

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

X_train_scaled = (X_train - X_mean)/X_std
X_test_scaled = (X_test - X_mean)/X_std

def knn_predict(X_train,y_train,new_data, k=3):
    predictions = []

    for point in new_data:
        distance = np.sqrt(np.sum((X_train - point) ** 2,axis=1))

        nearest_indices = np.argsort(distance)[:k]

        nearest_classes = y_train[nearest_indices]

        values,count = np.unique(nearest_classes,return_counts=True)

        prediction = values[np.argmax(count)]

        predictions.append(prediction)

    return np.array(predictions)

k = 3

test_prediction = knn_predict(X_train_scaled,y_train,X_test_scaled,k)

print('\n======TEST PREDICTIONS=======')
for actual,predicted in zip(y_test,test_prediction):
    print(f"Actual: {actual} | Predicted : {predicted}")

accuarcy = np.mean(test_prediction == y_test)

print('\n=======MODEL EVALUTION======')
print('Accuracy: ',round(accuarcy*100,2),'%')

new_customer = np.array([
    [38, 57000, 58]
])

new_customer_scaled = (new_customer - X_mean)/X_std

new_predicted = knn_predict(X_train_scaled,y_train,new_customer_scaled,k)[0]

print("\n====== NEW CUSTOMER PREDICTION ======")

print('Age : ',new_customer[0][0])
print('Annual income : ',new_customer[0][1])
print('Spending score : ',new_customer[0][2])

print(
    "K:",
    k
)

print(
    "Predicted class:",
    new_predicted
)

if new_predicted == 0:
    print("Customer is NOT INTERESTED")
else:
     print("Customer is  INTERESTED")
