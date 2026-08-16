import pandas as pd
import joblib
import streamlit as st

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="AI Customer Retention Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.sidebar.title("📋 Navigation")
st.sidebar.info(
    "Fill in the customer details, then click **🚀 Predict Customer Churn** to receive an AI prediction."
)
st.sidebar.markdown("---")

st.sidebar.markdown("### 📊 Model Information")
st.sidebar.write("**Algorithm:** Random Forest Classifier")
st.sidebar.write("**Accuracy:** 79.25%")
st.sidebar.write("**Dataset:** IBM Telco Customer Churn")
# =====================================
# Load Model and Encoders
# =====================================
model = joblib.load("models/churn_model.pkl")
encoders = joblib.load("models/encoders.pkl")

# =====================================
# Title
# =====================================
st.title("📊 AI Customer Retention Dashboard")

st.caption(
    "Predict customer churn using Machine Learning and receive intelligent retention recommendations."
)

st.caption(
    "Predict customer churn using Machine Learning and receive retention recommendations."
)

st.info(
    "🤖 Enter customer information below and click **Predict Customer Churn** "
    "to receive an AI prediction, risk level, and retention recommendations."
)

st.divider()

# =====================================
# Customer Information
# =====================================
st.header("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12
    )

with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

# =====================================
# Billing Information
# =====================================

st.divider()

st.header("💳 Billing Information")

bill1, bill2 = st.columns(2)

with bill1:

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

with bill2:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = tenure * monthly_charges

st.divider()

predict_button = st.button(
    "🚀 Predict Customer Churn",
    use_container_width=True,
    type="primary"
)
if predict_button:

    # Encode inputs
    gender = encoders["gender"].transform([gender])[0]
    partner = encoders["Partner"].transform([partner])[0]
    dependents = encoders["Dependents"].transform([dependents])[0]
    phone_service = encoders["PhoneService"].transform([phone_service])[0]
    multiple_lines = encoders["MultipleLines"].transform([multiple_lines])[0]
    internet_service = encoders["InternetService"].transform([internet_service])[0]
    online_security = encoders["OnlineSecurity"].transform([online_security])[0]
    online_backup = encoders["OnlineBackup"].transform([online_backup])[0]
    device_protection = encoders["DeviceProtection"].transform([device_protection])[0]
    tech_support = encoders["TechSupport"].transform([tech_support])[0]
    streaming_tv = encoders["StreamingTV"].transform([streaming_tv])[0]
    streaming_movies = encoders["StreamingMovies"].transform([streaming_movies])[0]
    contract = encoders["Contract"].transform([contract])[0]
    paperless = encoders["PaperlessBilling"].transform([paperless])[0]
    payment_method = encoders["PaymentMethod"].transform([payment_method])[0]

    senior = 1 if senior == "Yes" else 0
        # Create input data
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.header("📈 Prediction Results")
    st.caption("Results are generated using the trained Random Forest model.")
    

    if prediction == 1:

        st.error("🔴 Prediction: Customer IS likely to churn")
        st.toast("Customer is at risk of leaving!", icon="⚠️")
        

    else:

        st.success("🟢 Prediction: Customer is NOT likely to churn")
        st.toast("Customer is likely to stay 🎉", icon="✅")
        

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
    "Prediction",
    "🔴 Churn" if prediction == 1 else "🟢 No Churn"
    )

    with col2:
        st.metric(
    "Churn Probability",
    f"{probability:.2%}",
    delta=f"{probability * 100:.1f}%"
    )
        
    with col3:
        risk = (
            "High" if probability >= 0.70
            else "Medium" if probability >= 0.40
            else "Low"
    )
    risk_color = {
    "Low": "🟢 Low",
    "Medium": "🟡 Medium",
    "High": "🔴 High"
    }

    st.metric("Risk Level", risk_color[risk])

    # Risk Level
    if probability >= 0.70:

        st.warning("Risk Level: HIGH")

        st.subheader("Recommended Actions")

        st.markdown("✅ Offer a loyalty discount")
        st.markdown("📞 Contact the customer within 48 hours")
        st.markdown("👤 Assign a customer success representative")

    elif probability >= 0.40:

        st.info("🟡 Risk Level: MEDIUM")

        st.subheader("💡 Recommended Actions")

        st.markdown("📧 Send a promotional email")
        st.markdown("🎁 Offer a limited-time discount")
        st.markdown("📊 Monitor customer activity")

    else:

        st.success("🟢 Risk Level: LOW")

        st.subheader("💡 Recommended Actions")

        st.markdown("⭐ Continue providing excellent service")
        st.markdown("🏆 Reward customer loyalty")
        st.markdown("❤️ Send appreciation messages")