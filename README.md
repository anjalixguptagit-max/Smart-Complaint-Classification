# 📩 Smart Complaint Classification

A simple ML project that reads customer complaints and predicts the right complaint category and department.

## What it does

- Takes a customer complaint as input
- Predicts the complaint category
- Suggests the relevant department
- Gives a recommended action
- Shows important words from the complaint

## Categories

- Credit Card
- Credit Reporting
- Debt Collection
- Mortgages & Loans
- Retail Banking

## Model

We used:

- TF-IDF for converting text into numbers
- Linear SVM for classification
- Streamlit for the web app

### Model Result

**Test Accuracy: 89.14%**

**Macro F1 Score: 0.85**

The dataset contains **162,411 complaints**.

## How it works

```text
Customer Complaint
        ↓
      TF-IDF
        ↓
    Linear SVM
        ↓
Complaint Category
        ↓
Department + Recommended Action
```

## Run the Project

Clone the repository:

```bash
git clone https://github.com/anjalixguptagit-max/Smart-Complaint-Classification.git
cd Smart-Complaint-Classification
```

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

Run the app:

```bash
streamlit run app.py
```

## Example

**Input:**

```text
My credit card was charged for a transaction that I did not make.
```

**Output:**

```text
Category: Credit Card
Department: Credit Card Support
```

## Team

- Aashish Dhanwani
- Anjali Gupta

Made for our college hackathon ❤️
