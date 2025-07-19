# Customer Churn Prediction

This is a **Machine Learning project** developed as part of the **CODSOFT Internship Program**. It predicts whether a bank customer is likely to **exit (churn)** or stay, based on historical customer data.

## Project Summary

Customer churn is a critical metric for any business. Using this model, banks can proactively identify customers at risk of leaving and take preventive actions. This project uses a supervised ML approach with a **Random Forest Classifier** trained on structured banking data.

## Dataset

- **Name**: `Churn_Modelling.csv`
- **Rows**: 10,000
- **Features**: 14
- **Target**: `Exited` (1 = Churn, 0 = Not Churn)

## Technologies & Tools

- Python
- Pandas, NumPy
- Scikit-learn
- Jupyter / VS Code
- Git & GitHub

## Model Details

- **Algorithm**: Random Forest Classifier
- **Accuracy**: `~86.8%`
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix

### Classification Report:
           precision    recall  f1-score   support

       0       0.88      0.96      0.92      1607
       1       0.76      0.49      0.59       393

Accuracy                           0.87      2000

## Features Engineered

- `Gender` converted to binary
- `Geography` converted using one-hot encoding
- Removed irrelevant columns: `CustomerId`, `RowNumber`, `Surname`

## Sample Prediction

```python
# Sample Input
sample_input = pd.DataFrame([{
    'CreditScore': 650,
    'Gender': 1,
    'Age': 35,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 1,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000,
    'Geography_Germany': 0,
    'Geography_Spain': 1
}])

# Output
Predicted Churn (1 = Yes, 0 = No): 0

## Model File
The trained model is saved as churn_model.pkl using joblib.

## Demo

[ Demo Video Link (Coming Soon) ](#)

## Internship Info

Company: CodSoft

Domain: Machine Learning

Project: 3 of 3 (Final Project)

Duration: June–July 2025

## Author

**Arpit Kumar Singh**  
CodSoft Machine Learning Intern – July 2025  
GitHub: [@ArpitKS](https://github.com/ArpitKS)
