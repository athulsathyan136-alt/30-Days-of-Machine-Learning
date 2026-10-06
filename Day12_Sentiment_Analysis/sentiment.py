import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/reviews.csv')

print('\n=======DATASET==========')
print(df)

X = df['review']

y = df['sentiment']

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print('======DATA SPLIT=========')
print('Trainig sample ',len(X_train))
print('Testing sample ',len(X_test))

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english'
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print('\n====TEXT CONVERSION======')
print('Trainig matrix shape ',X_train_tfidf.shape)
print('Testing matrix shape ',X_test_tfidf.shape)
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf,y_train)

print('\n=======MODEL TRAINED========')

predictions = model.predict(X_test_tfidf)

print('\n=======TEST PREDICTIONS=========')

for reviews, actual, predicted in zip(X_test,y_test,predictions):
    print('Reviews: ',reviews)
    print(f'Actual: {actual} | Predicted: {predicted} ')
    print('-'*60)

accuarcy = accuracy_score(y_test,predictions)

print('\n=====MODEL EVALUTION========')
print('Accuracy ',round(accuarcy * 100 ,2),'%')

new_reviews = [
    "I absolutely love this product",
    "This product is awful",
    "The quality is excellent",
    "I am very disappointed with this service",
    "The app is simple and useful"
]

new_reviews_tfidf = vectorizer.transform(new_reviews)

new_prediction = model.predict(new_reviews_tfidf)

print('\n=====NEW REVIEW PREDICTIONS=======')
for review , predictions in zip(new_reviews,new_prediction):
    print('Review ',review)
    print('Prediction: ',predictions)
    print('-'*60)