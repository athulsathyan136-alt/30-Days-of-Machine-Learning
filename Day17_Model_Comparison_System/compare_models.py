from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
iris = load_iris()

X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


scaler = StandardScaler
x_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression':LogisticRegression(max_iter=1000),'K-nearest Neighbors':KNeighborsClassifier(n_neighbors=5),
    'Decision Tree':DecisionTreeClassifier(random_state=42),'Random forest ': RandomForestClassifier(n_estimators=100,random_state=42)

}

print("=" * 50)
print("       MODEL COMPARISON SYSTEM")
print("=" * 50)


print("====== DATA PREPARATION ======")
print('Total samples: ',len(X))
print('Training samples: ',len(X_train))
print('Testing samples: ',len(X_test))
print('Number of features: ',X.shape[1])
print("Training data prepared successfully!")

result = {}

for model_name ,model in models.items():
    if model_name in ['Logistic Regression','K-nearest Neighbors']:
        model.fit(x_train_scaled,y_train)
        prediction = model.predict(X_test_scaled)

    else:
        model.fit(X_train,y_train)
        prediction = model.predict(X_test)


    accuracy = accuracy_score(y_test,prediction)

    result[model_name] = accuracy

    print(f'\nModel: {model_name}')
    print(f'Accuarcy: {accuracy *100:.2f} %')

print("\n" + "=" * 50)
print("          MODEL COMPARISON RESULTS")
print("=" * 50)