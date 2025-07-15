# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load Dataset
data = pd.read_csv('fraudTrain.csv')
data = data.sample(n=100000, random_state=42)

# Explore Basic Info
print("Dataset shape:", data.shape)
print("\nFirst 5 rows:\n", data.head())

# Features & Target
print("\n Columns in dataset:\n", data.columns)
X = data.drop(columns=[
    'is_fraud', 'trans_date_trans_time', 'cc_num', 'merchant',
    'category', 'first', 'last', 'gender', 'street', 'city', 'state',
    'zip', 'job', 'dob', 'trans_num'
])

y = data['is_fraud']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier()
print("\n Training model, please wait...\n")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\n Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\n Sample Predictions (Clean):")
actual_list = list(y_test.iloc[:10])
predicted_list = list(map(int, y_pred[:10]))

print("Actual:    ", actual_list)
print("Predicted: ", predicted_list)

print("\n Showing fraud samples only:")
fraud_indexes = y_test[y_test == 1].index[:5]
for idx in fraud_indexes:
    actual = y_test.loc[idx]
    predicted = model.predict(X_test.loc[[idx]])[0]
    print(f"Actual: {actual} → Predicted: {predicted}")


import joblib

# Save model
joblib.dump(model, "creditcard_model.pkl")
print("\n Model saved as creditcard_model.pkl \n")


