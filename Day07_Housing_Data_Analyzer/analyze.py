import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/housing_data.csv')

print('======DATASET=========')
print(df)

print('=======DATASET INFO=======')
df.info()

print('=======MISSING VALUES=======')
print(df.isnull().sum())

print('=======DUPLICATES=======')
print(df.duplicated().sum())

print('====== STATISTICS=======')
print(df.describe())

print('=======AVERAGE=======')
print('Average area: ',round(df["area_sqft"].mean(),2),'sqft')
print('Average bedrooms: ',round(df["bedrooms"].mean(),2))
print('Average bathrooms: ',round(df["bathrooms"].mean(),2),)
print('Average house age: ',round(df["age_years"].mean(),2),'year')
print('Average price: ',round(df["price_lakh"].mean(),2),'lakh')

print('\n=======CORRELATION===========')

correlation = df.corr(numeric_only=True)

print(correlation)

print('\n=========CORRELATION WITH PRICE===========')
print(correlation["price_lakh"].sort_values(ascending=False))

print('\n=======HOUSES ABOVE 70 LAKH==============')
print(df[df["price_lakh"]>70])

print('\n=======HOUSES ABOVE 1400 SQFT ==============')
print(df[df["area_sqft"]>1400])

print('\n=======MOST EXPENSIVE HOUSE==============')
most = df.loc[df['price_lakh'].idxmax()]
print(most)

plt.figure(figsize=(8,5))

plt.scatter(
    df["area_sqft"],
    df["price_lakh"]
)
plt.xlabel('Area (sqft)')
plt.ylabel('Price (lakh)')
plt.title('House Area vs Price')

plt.tight_layout()

plt.savefig('plots/area_vs_price.png')
plt.show()


plt.figure(figsize=(8,5))

plt.scatter(
    df["age_years"],
    df["price_lakh"]
)
plt.xlabel('House Age (years)')
plt.ylabel('Price (lakh)')
plt.title('House Age vs Price')

plt.tight_layout()

plt.savefig('plots/age_vs_price.png')
plt.show()

print('=======COMPLETED============')
print('Saving....................')