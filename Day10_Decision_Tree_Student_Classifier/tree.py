import pandas as pd
import numpy as np

df = pd.read_csv('data/students.csv')

print('=====DATASET=======')
print(df)

X = df[['hours_studied','attendance','assignments']].values

y = df['passed'].values

np.random.seed(42)

indices = np.random.permutation(len(X))

split = int(len(X)*0.8)

train_indices = indices[:split]
test_indices = indices[split:]

X_train = X[train_indices]
y_train = y[train_indices]

X_test = X[test_indices]
y_test = y[test_indices]

print('=====DATA SPLIT========')
print('Training samples: ',len(X_train))
print('Testing samples: ',len(X_test))

def gini(values):
    if len(values) == 0:
        return 0

    classes,count = np.unique(values,return_counts=True)

    probabilites = count / len(values)

    return 1 - np.sum(probabilites ** 2)

best_feature = None
best_threshold = None
best_gini = float("inf")

for feature_index in range(X_train.shape[1]):
    thresholds = np.unique(X_train[:,feature_index])

    for thershold in thresholds:
        left_mask = (X_train[:,feature_index] <= thershold)
        right_mask = (X_train[:,feature_index] > thershold)

        left_y = y_train[left_mask]
        right_y = y_train[right_mask]

        if len(left_mask) == 0 or len(right_y) == 0:
            continue

        left_gini = gini(left_y)
        right_gini = gini(right_y)

        weighted_gini = (
            (len(left_y) / len(y_train)) *left_gini + (len(right_y) / len(y_train)) * right_gini)

        if weighted_gini < best_gini:
            best_gini = weighted_gini
            best_feature = feature_index
            best_threshold = thershold

featues_name = ['hours_studied','attendance','assignments']

print('\n======DECISION TREE========')
print('Best feature: ',featues_name[best_feature])
print('Best threshold: ',best_threshold)
print('Gini: ',round(best_gini,4))

def predict_point(point):
    if point[best_feature] <= best_threshold:
        group = y_train[X_train[:,best_feature] <= best_threshold]

    else:
        group = y_train[X_train[:,best_feature] > best_threshold]

    classes , count = np.unique(group,return_counts=True)

    return classes[np.argmax(count)]

test_predictions = np.array([predict_point(point) for point in X_test])

print('\n=======TEST PREDICTIONS=========')

for actual, predicted in zip(
    y_test,
    test_predictions
):
    print(
        f"Actual: {actual} | "
        f"Predicted: {predicted}"
    )


# -----------------------------
# ACCURACY
# -----------------------------

accuracy = np.mean(
    test_predictions == y_test
)

print("\n====== MODEL EVALUATION ======")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# -----------------------------
# NEW STUDENT
# -----------------------------

new_student = np.array([
    [6, 84, 8]
])

new_prediction = predict_point(
    new_student[0]
)

print("\n====== NEW STUDENT PREDICTION ======")

print(
    "Hours studied:",
    new_student[0][0]
)

print(
    "Attendance:",
    new_student[0][1]
)

print(
    "Assignments:",
    new_student[0][2]
)

print(
    "Predicted class:",
    new_prediction
)

if new_prediction == 0:

    print("Student will likely FAIL")

else:

    print("Student will likely PASS")
