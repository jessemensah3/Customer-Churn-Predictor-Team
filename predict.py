import joblib

# Load the trained model
model = joblib.load("models/churn_model.pkl")

print("Model loaded successfully!")