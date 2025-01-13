import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# Load the dataset
file_path = 'data/spam.csv'
data = pd.read_csv(file_path, encoding='latin-1')

# Preprocess the dataset
data = data[['v1', 'v2']]
data.columns = ['Label', 'Message']
data['Label'] = data['Label'].map({'ham': 0, 'spam': 1})

# Display first few rows of cleaned dataset
print("First few rows of cleaned dataset:")
print(data.head())
print("\nDataset shape:", data.shape)

# Show example of tokenization
example_tokens = word_tokenize(data['Message'].iloc[0])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data['Message'], data['Label'], test_size=0.2, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions and calculate metrics
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display accuracy and confusion matrix
print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))
print("\nConfusion Matrix:")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Save the model and vectorizer
model_path = 'model/sms_model.pkl'
vectorizer_path = 'model/vectorizer.pkl'

with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)

with open(vectorizer_path, 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print(f"\nModel and vectorizer saved:\n- {model_path}\n- {vectorizer_path}")