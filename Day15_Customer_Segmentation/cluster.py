import pandas as pd

import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('data/customers.csv')

print('\n======DATASET===========')
print(df)

X = df[['age',
        'annual_income',
        'spending_score']]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df['cluster'] = model.fit_predict(X_scaled)

print('\n=======CUSTOMERS  GROUPS=======')
print(df)

print("\n====== CUSTOMERS PER CLUSTER ======")
print(df['cluster'].value_counts().sort_index())

centers = scaler.inverse_transform(model.cluster_centers_)

centers_df = pd.DataFrame(centers,columns=['age','annual_income','spending_score'])

print("\n====== CLUSTER CENTERS ======")
print(centers_df.round(2))

plt.figure(figsize=(9,5))

plt.scatter(df['annual_income'],
            df['spending_score'],
            c=df['cluster'],
            cmap='viridis',
            s=80)

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customers Segmentation using K-means')
plt.colorbar(label='Cluster')

plt.tight_layout()
plt.savefig('plot/customer_segments.png')
plt.show()

new_customer = pd.DataFrame({
    "age": [35],
    "annual_income": [52000],
    "spending_score": [65]
})
new_customer_scaled = scaler.transform(new_customer)

new_cluster = model.predict(new_customer_scaled)

print("\n====== NEW CUSTOMER ======")

print("Age:", new_customer["age"].iloc[0])
print("Annual income:", new_customer["annual_income"].iloc[0])
print("Spending score:", new_customer["spending_score"].iloc[0])

print("Assigned cluster:", new_cluster[0])

print("\n====== COMPLETED ======")
print("Graph saved to plots/customer_segments.png")