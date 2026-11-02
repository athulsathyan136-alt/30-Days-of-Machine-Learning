from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "customer_segmentation.joblib"

package = joblib.load(MODEL_PATH)

model = package["model"]
scaler = package["scaler"]
features = package["features"]

print("\nCUSTOMER SEGMENTATION PREDICTION")
print("=" * 40)

customer = {
    "Age": float(input("Enter customer age: ")),
    "AnnualIncome": float(input("Enter annual income: ")),
    "SpendingScore": float(input("Enter spending score (1-100): ")),
    "PurchaseFrequency": float(input("Enter monthly purchase frequency: ")),
    "AverageOrderValue": float(input("Enter average order value: ")),
    "WebsiteVisits": float(input("Enter monthly website visits: ")),
}

if not 18 <= customer["Age"] <= 75:
    raise ValueError("Age must be between 18 and 75.")

if not 1 <= customer["SpendingScore"] <= 100:
    raise ValueError("Spending score must be between 1 and 100.")

if any(customer[key] <= 0 for key in [
    "AnnualIncome",
    "PurchaseFrequency",
    "AverageOrderValue",
    "WebsiteVisits"
]):
    raise ValueError("Income, frequency, order value and visits must be positive.")

customer_df = pd.DataFrame([customer], columns=features)

scaled_customer = scaler.transform(customer_df)

cluster = model.predict(scaled_customer)[0]

print("\nPREDICTION RESULT")
print("=" * 40)
print(f"Assigned customer cluster: {cluster}")

print(
    "\nNote: Cluster numbers are mathematical group labels, "
    "not rankings or predefined customer types."
)