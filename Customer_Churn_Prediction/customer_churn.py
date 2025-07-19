# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Load Dataset
data = pd.read_csv("Churn_Modelling.csv")
print("Dataset shape:", data.shape)
print("\nFirst 5 rows:\n", data.head())

# Drop irrelevant columns
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Encode categorical features
data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})
data = pd.get_dummies(data, columns=['Geography'], drop_first=True)

# Features and Target
X = data.drop('Exited', axis=1)
y = data['Exited']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("\nTraining model, please wait...")
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model
joblib.dump(model, "churn_model.pkl")
print("\nModel saved as churn_model.pkl")

# Load model and test prediction
import joblib

model = joblib.load("churn_model.pkl")

# Sample Customer input
sample_input = pd.DataFrame([{
    'CreditScore': 619,
    'Gender': 0,  # Female = 0, Male = 1
    'Age': 42,
    'Tenure': 2,
    'Balance': 0.00,
    'NumOfProducts': 1,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 101348.88,
    'Geography_Germany': 0,
    'Geography_Spain': 0
}])

# Predict
prediction = model.predict(sample_input)[0]
print("\nSample Prediction Demo")
print("\nPredicted Churn (1 = Yes, 0 = No):", prediction)
print("\n")

