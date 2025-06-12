# 🕵️ Enron Fraud Detection with Machine Learning (Python)

**Project Type:** Supervised Machine Learning  
**Tools:** Python, scikit-learn, pandas, matplotlib  
**Dataset:** Enron Email + Financial Dataset (~150 individuals, 20+ features)

---

## 📌 Project Overview

This project investigates financial fraud at Enron using machine learning. By analyzing both financial and email metadata, we aim to build a binary classification model that identifies **Persons of Interest (POIs)** — individuals who were indicted, settled, or testified in the corporate fraud investigation.

This public dataset was made available following the U.S. federal investigation into Enron’s collapse. The project includes:
- Data cleaning and exploration
- Feature engineering and selection
- Classifier training and tuning
- Model evaluation with precision, recall, and F1-score

---

## 📎 Acknowledgments

This project was completed as part of the **Udacity Machine Learning Engineer Nanodegree**.  
Original dataset and starter code provided by **[Udacity's open-source curriculum](https://github.com/udacity/ud120-projects)**.

Additional inspiration and feature engineering ideas referenced from:
- [zelite/Identify-fraud-from-enron-email](https://github.com/zelite/Identify-fraud-from-enron-email)
- [jasminej90/dand5-identity-fraud-from-enron-email](https://github.com/jasminej90/dand5-identity-fraud-from-enron-email)

---

## ⚖️ License

This repository is provided under the **[MIT License](LICENSE)**.  
You are free to use, modify, and share this code for personal or educational purposes.

---

## 🎯 Why This Project Matters

Corporate fraud detection is a high-impact application of machine learning. This project highlights:
- The importance of **domain knowledge** in feature selection
- The challenge of **imbalanced datasets** and small sample sizes
- The need for **interpretable models** in high-stakes contexts like fraud and finance

---

## 🗃️ Dataset Details

- 📁 Source: [Udacity Enron Dataset](https://github.com/udacity/ud120-projects)
- 👥 ~150 individuals from Enron
- 💼 ~20 financial and email communication features:
  - salary, bonus, total payments, exercised stock options, emails to/from POIs, etc.
- 🏷️ Target variable: `poi` (1 = Person of Interest, 0 = Not)

Each row represents an individual employee, with features engineered from financial records and communication metadata.

---

## 🧰 Tools & Techniques Used

- **Python**
  - `scikit-learn` for machine learning
  - `pandas` for data manipulation
  - `matplotlib` for visualizations
- Feature selection techniques:
  - `SelectKBest`, decision tree feature importances
- Classifiers:
  - Gaussian Naive Bayes, Decision Trees, SVM
- Model tuning:
  - `GridSearchCV`, cross-validation
