# AI Customer Retention Dashboard

## Overview

The AI Customer Retention Dashboard is a machine learning application developed using Python and Streamlit. The system predicts whether a telecommunications customer is likely to churn based on customer information and subscription details.

The application also provides customer retention recommendations based on the predicted level of churn risk.

---

## Features

- Customer churn prediction
- Churn probability estimation
- Risk classification (Low, Medium, High)
- Retention recommendations
- Interactive Streamlit dashboard
- Machine learning model trained using the IBM Telco Customer Churn dataset

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

---

## Dataset

IBM Telco Customer Churn Dataset

Features include:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Contract
- Monthly Charges
- Total Charges

---

## Machine Learning Model

Algorithm:
- Random Forest Classifier

Model Accuracy:
- 79.25%

---

## Installation

Clone the repository

```bash
git clone <repository-url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Project Structure

```
Customer-Churn-Predictor/

│── app.py
│── train_model.py
│── preprocess.py
│── predict.py
│── requirements.txt
│── README.md

├── data/
├── models/
```

---

## Authors

Group Project

Ashesi University

---

## License

For educational purposes only.