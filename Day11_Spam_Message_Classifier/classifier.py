import pandas as  pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/messages.csv')

print('======DATASET=======')
print(df)

X = df['message']
y = df['label']


X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print('\n=====DATA SPLIT========')
print('Training sample: ',len(X_train))
print('Testing sample: ',len(X_test))

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words = 'english'
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print('\n=====TEXT CONVERSION========')
print('Training matrix shape: ',X_train_tfidf.shape)
print('Testing matrix shape: ',X_test_tfidf.shape)

model = MultinomialNB()

model.fit(X_train_tfidf,y_train)

print('\n =========MODEL TRAINED=========')
predictions = model.predict(X_test_tfidf)

print('\n=====TEST PREDICTION=========')

for message , actual ,predicted in zip(X_test,y_test,predictions):
    print(f'Message: {message}')
    print(f'Actual :{actual} | Predicted: {predicted}')
    print('-'*60)

accuracy = accuracy_score(y_test,predictions)

print('\n=========MODEL EVALUATION============')
print('ACCURACY: ',round(accuracy * 100 ,2),'%')

new_messages = [
    "Congratulations you won a free cash prize",
    "Can you send me the meeting notes",
    "Claim your free reward now",
    "Are you coming to class tomorrow"
]

new_messages_tfidf = vectorizer.transform(new_messages)

new_predictions = model.predict(new_messages_tfidf)

print('\n========NEW MESSAGE PREDICTIONS===========')

for message,predictions in zip(new_messages,new_predictions):
    print(f'Message: {message}')
    print(f'Prediction: {predictions}')
    print('='*60)