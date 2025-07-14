# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']  # Rename columns for simplicity

# Preprocess labels (spam = 1, ham = 0)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Show results
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📌 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

import joblib

# Save the model
joblib.dump(model, "spam_classifier_model.pkl")

# Save the vectorizer (used to convert messages)
joblib.dump(vectorizer, "vectorizer.pkl")

# Load model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Take user input
message = input("Enter a message: ")

# Predict
msg_vec = vectorizer.transform([message])
result = model.predict(msg_vec)

if result[0] == 1:
    print("This is SPAM")
else:
    print("This is NOT SPAM")

