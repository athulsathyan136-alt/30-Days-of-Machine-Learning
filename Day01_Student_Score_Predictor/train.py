import pandas as pd


# 1. Load dataset
df = pd.read_csv("data/student_scores.csv")

print("===== DATASET =====")
print(df)


# 2. Get X and y
X = df["hours_studied"].values
y = df["exam_score"].values


# 3. Calculate means
x_mean = X.mean()
y_mean = y.mean()


# 4. Calculate Linear Regression parameters
numerator = ((X - x_mean) * (y - y_mean)).sum()
denominator = ((X - x_mean) ** 2).sum()

slope = numerator / denominator
intercept = y_mean - (slope * x_mean)


print("\n===== MODEL =====")
print("Slope:", slope)
print("Intercept:", intercept)


# 5. Make predictions
predictions = slope * X + intercept

print("\n===== PREDICTIONS =====")

for hours, actual, predicted in zip(X, y, predictions):
    print(
        f"Hours: {hours} | "
        f"Actual: {actual} | "
        f"Predicted: {predicted:.2f}"
    )


# 6. Predict a new student
new_hours = 7.5
new_prediction = slope * new_hours + intercept

print("\n===== NEW STUDENT =====")
print("Hours studied:", new_hours)
print("Predicted score:", round(new_prediction, 2))