import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
# -----------------------------
# Load the dataset
# -----------------------------
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Remove customerID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove missing values
df.dropna(inplace=True)

# Encode categorical columns
encoders = {}

for column in df.select_dtypes(include=["object", "string"]).columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column].astype(str))
    encoders[column] = encoder

# Split features and target
X = df.drop("Churn", axis=1)
print(X.columns.tolist())
y = df["Churn"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train the model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
# -----------------------------
# Make predictions
# -----------------------------
predictions = model.predict(X_test)

# -----------------------------
# Evaluate the model
# -----------------------------
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy:.2%}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save the trained model
joblib.dump(model, "models/churn_model.pkl")

# Save the encoders
joblib.dump(encoders, "models/encoders.pkl")

print("\nModel and encoders saved successfully!")