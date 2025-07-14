# Spam SMS Detection – ML Project

This is a machine learning project built as part of the **CodSoft Machine Learning Internship**. The goal is to build a model that detects whether an SMS message is **SPAM or NOT SPAM** using Natural Language Processing (NLP).

---

## Project Overview

- **Algorithm Used**: Multinomial Naive Bayes
- **Technique**: TF-IDF Vectorization
- **Dataset**: [SMS Spam Collection Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Language**: Python
- **Libraries**: `pandas`, `scikit-learn`, `joblib`

---

## How It Works

1. **Dataset Preprocessing**
   - Removed unnecessary columns
   - Converted labels (`ham`, `spam`) to numeric (`0`, `1`)

2. **Text Vectorization**
   - Used TF-IDF to convert text to vectors

3. **Model Training**
   - Trained a Naive Bayes model on training data
   - Achieved high accuracy on test data

4. **CLI Input**
   - User can enter a message → Model predicts spam or not

---

## Project Structure

SpamSMSDetection/
├── spam.csv
├── spam_sms_detection.py
├── spam_classifier_model.pkl
├── vectorizer.pkl
├── README.md

---

## Demo

[ Demo Video Link (Coming Soon) ](#)

---

## Author

**Arpit Kumar Singh**  
CodSoft Machine Learning Intern – July 2025  
GitHub: [@ArpitKS](https://github.com/ArpitKS)

---

## Tags

`#MachineLearning` `#CodSoftInternship` `#SpamDetection` `#Python` `#NLP` `#MLProject`


