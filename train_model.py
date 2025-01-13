import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load the dataset
file_path = 'data/spam.csv'
data = pd.read_csv(file_path, encoding='latin-1')

# Preprocess the dataset
data = data[['v1', 'v2']]
data.columns = ['Label', 'Message']
data['Label'] = data['Label'].map({'ham': 0, 'spam': 1})

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data['Message'], data['Label'], test_size=0.2, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Save the model and vectorizer
model_path = 'model/sms_model.pkl'
vectorizer_path = 'model/vectorizer.pkl'

with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)

with open(vectorizer_path, 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print(f"Model and vectorizer saved:\n- {model_path}\n- {vectorizer_path}")
