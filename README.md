# 📩 Smart Complaint Classification

An AI-powered complaint classification system that automatically analyzes customer complaints, predicts the most relevant complaint category, routes the complaint to the appropriate department, and recommends the next action.

Built for the Smart Complaint Classification hackathon problem statement.

---

## 🚀 Project Overview

Organizations receive a large number of customer complaints every day. Manually reading, categorizing, and forwarding these complaints to the correct department can be time-consuming and inefficient.

**Smart Complaint Classification** automates this process using Natural Language Processing (NLP) and Machine Learning.

The system takes a customer's complaint as input and automatically:

1. Analyzes the complaint text
2. Converts text into numerical features using **TF-IDF**
3. Predicts the complaint category using **Linear SVM**
4. Routes the complaint to the appropriate department
5. Generates a recommended action
6. Extracts important terms from the complaint

---

## 🎯 Problem Statement

### Smart Complaint Classification

Organizations receive many customer complaints that need to be manually sorted into categories and departments.

The goal of this project is to build a machine learning system that can:

- Analyze customer complaints
- Predict the appropriate complaint category
- Route the complaint to the relevant department
- Provide useful information for faster complaint handling

---

## ✨ Features

- 🤖 Automated complaint classification
- 🧠 NLP-based text analysis
- 🔤 TF-IDF feature extraction
- ⚡ Linear SVM classification
- 🎯 5 complaint categories
- 🏢 Automatic department routing
- 📌 Recommended action generation
- 🔑 Key complaint terms detection
- 📊 Model performance metrics
- 🌐 Interactive Streamlit web interface
- 🎨 Modern AI-themed UI

---

## 📂 Complaint Categories

| Category | Department |
|---|---|
| Credit Card | Credit Card Support |
| Credit Reporting | Credit Reporting Team |
| Debt Collection | Debt Collection Team |
| Mortgages & Loans | Loans & Mortgage Department |
| Retail Banking | Retail Banking Support |

---

## 🧠 Machine Learning Approach

### 1. Text Preprocessing

Customer complaint text is processed and prepared for machine learning.

### 2. TF-IDF

The complaint text is converted into numerical features using:

**TF-IDF — Term Frequency-Inverse Document Frequency**

The vectorizer uses:

- Maximum Features: `50,000`
- N-gram Range: `(1, 2)`

This allows the model to capture both individual words and meaningful two-word phrases.

### 3. Classification Model

A **Linear Support Vector Machine (Linear SVM)** is used for complaint classification.

Linear SVM was selected because it performs well for high-dimensional text classification problems and provides efficient prediction.

---

## 📊 Model Performance

| Metric | Score |
|---|---:|
| Test Accuracy | **89.14%** |
| Macro F1 Score | **0.85** |
| Weighted F1 Score | **0.89** |
| Training Samples | **129,928** |
| Testing Samples | **32,483** |
| Categories | **5** |

---

## 📈 Dataset

The project uses a customer complaint dataset containing complaint narratives and product/category information.

After removing missing complaint narratives and unnecessary columns:

**162,411 complaint records** were used.

### Dataset Distribution

| Category | Complaints |
|---|---:|
| Credit Reporting | 91,172 |
| Debt Collection | 23,148 |
| Mortgages & Loans | 18,990 |
| Credit Card | 15,566 |
| Retail Banking | 13,535 |
| **Total** | **162,411** |

The dataset was divided using an **80/20 stratified train-test split**.

---

## 🏗️ System Architecture

```text
                Customer Complaint
                        │
                        ▼
                Text Preprocessing
                        │
                        ▼
                     TF-IDF
                        │
                        ▼
                  Linear SVM
                        │
                        ▼
              Complaint Category
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
      Department Routing      Confidence Score
            │
            ▼
      Recommended Action
            │
            ▼
          User
```

---

## 🖥️ Application Interface

The application is built using **Streamlit** and provides an interactive interface where users can enter a customer complaint and instantly receive:

- Predicted category
- Confidence score
- Recommended department
- Recommended action
- Key terms detected

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Machine Learning
- Scikit-learn
- Linear SVM
- TF-IDF

### Data Processing
- Pandas
- NumPy

### Web Interface
- Streamlit

### Model Serialization
- Joblib

### Visualization / UI
- HTML
- CSS
- Streamlit

---

## 📁 Project Structure

```text
Smart-Complaint-Classification/
│
├── assets/
│   └── background.png
│
├── data/
│   └── complaints_processed.csv
│
├── app.py
├── model.py
├── model.pkl
├── tfidf.pkl
├── metrics.json
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/anjalixguptagit-max/Smart-Complaint-Classification.git
```

Move into the project directory:

```bash
cd Smart-Complaint-Classification
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧪 Example

### Input

```text
My credit card was charged for a transaction that I did not make.
```

### Output

```text
Predicted Category:
Credit Card

Recommended Department:
Credit Card Support

Recommended Action:
Forward the complaint to Credit Card Support
for transaction verification.
```

---

## 📌 Why This Project?

Traditional complaint handling requires employees to manually read and categorize large numbers of complaints.

This system helps automate the initial triaging process by providing:

- Faster classification
- Consistent categorization
- Automatic department routing
- Reduced manual effort
- Action-oriented complaint handling

---

## 🔮 Future Improvements

Possible future improvements include:

- Real-time complaint dashboard
- Multi-language complaint support
- Advanced transformer-based models
- Complaint priority/severity prediction
- Duplicate complaint detection
- Complaint trend analysis
- Integration with CRM/helpdesk systems
- Feedback-based model retraining
- Explainable AI for classification decisions

---

## 👥 Team

### Hackathon Project

**Team Members:**

- Aashish Dhanwani
- Anjali Gupta

---

## 📜 License

This project was developed as a hackathon prototype for educational and demonstration purposes.
