# Customer Churn Prediction Using Random Forest  
### A Data-Driven Retention Strategy

## 📌 Project Overview
Customer churn is a critical challenge for subscription-based and service-oriented businesses. This project focuses on predicting whether a customer is likely to churn using a **Random Forest Classifier**, enabling organizations to take **proactive retention actions**.

By analyzing customer demographics, account information, and service usage patterns, this model helps identify high-risk customers and supports **data-driven decision-making**.

---

## 🎯 Problem Statement
Businesses lose revenue when customers discontinue services.  
The goal of this project is to:
- Predict **customer churn (Yes/No)** accurately
- Identify **key factors influencing churn**
- Assist businesses in **reducing churn through targeted strategies**

---

## 📊 Dataset Description
The dataset contains customer-level information commonly found in telecom and subscription services.

**Key Features Include:**
- Customer demographics (gender, senior citizen, dependents)
- Account information (contract type, tenure, payment method)
- Service usage (internet service, streaming, tech support)
- Billing details (monthly charges, total charges)

**Target Variable:**
- `Churn` → Yes / No

---

## 🛠️ Technologies & Tools Used
- **Programming Language:** Python  
- **Libraries:**
  - pandas, numpy – data processing
  - matplotlib, seaborn – data visualization
  - scikit-learn – model building & evaluation
  - pickle – model serialization
- **Framework:** Streamlit (for interactive UI)

---

## 🔄 Project Workflow
1. **Data Loading & Exploration**
2. **Data Cleaning & Preprocessing**
   - Handling missing values
   - Encoding categorical features
   - Feature scaling (if required)
3. **Train-Test Split**
4. **Model Building**
   - Random Forest Classifier
5. **Model Evaluation**
   - Accuracy
   - Precision, Recall, F1-Score
   - Confusion Matrix
6. **Feature Importance Analysis**
7. **Model Deployment**
   - Streamlit web application
   - Pickle file for trained model

---

## 🤖 Why Random Forest?
Random Forest was chosen because it:
- Handles **non-linear relationships** effectively
- Works well with **mixed data types**
- Reduces overfitting using ensemble learning
- Provides **feature importance insights**

---

## 📈 Model Performance
- Achieves **high classification accuracy**
- Balanced precision and recall for churn prediction
- Robust performance on unseen data

*(Exact metrics may vary depending on dataset split and tuning.)*

---

## 🖥️ Streamlit Web Application
The project includes an interactive Streamlit dashboard where users can:
- Input customer details
- Predict churn probability instantly
- View visual insights (tenure vs churn, contract impact, charges vs churn)

---

## 📂 Project Structure
