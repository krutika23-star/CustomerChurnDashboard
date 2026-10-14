import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

MODEL_PATH = "churn_model.pkl"
DATA_PATH = "customer_churn.csv"

@st.cache_resource
def load_model():
    import os
    import gdown
    import joblib

    model_path = "churn_model.pkl"
    file_id = "1l7KLmfvGhENbzozphld5a3FcP4ViIokj"

    if not os.path.exists(model_path):
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            model_path,
            quiet=False
        )

    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        st.error("Model download failed. Check Google Drive sharing permissions.")
        st.stop()

    return joblib.load(model_path)
    
@st.cache_data
def load_data(path):
    import os
    import pandas as pd

    if not os.path.exists(path):
        st.error(f"Dataset file not found: {path}")
        st.stop()

    return pd.read_csv(path)

st.title("📊 Customer Churn Prediction")
st.caption("Machine Learning CA4 Project | Customer Retention Analytics")

model = load_model()
df = load_data(DATA_PATH)

expected_features = list(getattr(model, "feature_names_in_", []))
if not expected_features:
    expected_features = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges"
    ]

target_candidates = [
    c for c in df.columns
    if str(c).lower().replace("_", "").replace(" ", "")
    in {"churn", "churnlabel", "exited", "attrition"}
]
target_col = target_candidates[0] if target_candidates else None

def positive_churn_mask(series):
    return series.astype(str).str.strip().str.lower().isin(
        ["yes", "1", "true", "churned", "left"]
    )

# Metrics from a labeled dataset are descriptive, not held-out test metrics.
# Held-out evaluation metrics should be added only if the test set is exported.
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Feature Analysis", "Model Evaluation", "Live Prediction"]
)

if page == "Overview":
    st.subheader("Dashboard Overview")
    if target_col:
        churn_mask = positive_churn_mask(df[target_col])
        churn_rate = churn_mask.mean() * 100
        churn_count = int(churn_mask.sum())
    else:
        churn_rate = None
        churn_count = None

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Customers in Dataset Marked Churn", f"{churn_count:,}" if churn_count is not None else "Target column not found")
    c3.metric("Dataset Churn Rate", f"{churn_rate:.1f}%" if churn_rate is not None else "Unavailable")

    st.info(
        "Dataset overview metrics describe the CSV. Model evaluation metrics should come from a held-out test set."
    )

    if target_col:
        counts = df[target_col].astype(str).value_counts().rename_axis("Customer Status").reset_index(name="Customers")
        fig = px.pie(
            counts, names="Customer Status", values="Customers",
            title="Customer Churn Distribution", hole=0.45
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No churn target column was detected in the CSV.")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)



elif page == "Feature Analysis":
    st.subheader("Interactive Feature Analysis")

    feature_options = [
        c for c in df.columns
        if c not in [target_col, "customerID", "customerId", "CustomerID"]
    ]

    if not feature_options:
        st.warning("No feature columns are available.")
    else:
        feature = st.selectbox("Choose a feature", feature_options)
        series = df[feature]

        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 20:
            if target_col:
                fig = px.histogram(
                    df, x=feature, color=target_col,
                    marginal="box", barmode="overlay",
                    title=f"{feature} Distribution by Churn Status"
                )
            else:
                fig = px.histogram(
                    df, x=feature,
                    title=f"{feature} Distribution"
                )
        else:
            plot_df = df.copy()
            plot_df[feature] = plot_df[feature].fillna("Missing").astype(str)

            if target_col:
                counts = (
                    plot_df.groupby([feature, target_col], dropna=False)
                    .size().reset_index(name="Customers")
                )
                fig = px.bar(
                    counts, x=feature, y="Customers", color=target_col,
                    barmode="group",
                    title=f"{feature} by Churn Status"
                )
            else:
                counts = plot_df[feature].value_counts().reset_index()
                counts.columns = [feature, "Customers"]
                fig = px.bar(
                    counts, x=feature, y="Customers",
                    title=f"{feature} Distribution"
                )

        st.plotly_chart(fig, use_container_width=True)


elif page == "Model Evaluation":
    st.subheader("Model Evaluation")

    metrics = [
        ("Accuracy", 78.28),
        ("Precision", 61.81),
        ("Recall", 47.59),
        ("F1-score", 53.78),
    ]

    cols = st.columns(4)
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, f"{value:.2f}%")

    st.markdown("### Interpretation")
    st.write(
        "Accuracy measures overall correctness. Precision measures "
        "the reliability of positive churn predictions. Recall measures "
        "how many actual churners were identified. F1-score combines "
        "precision and recall."
    )

    st.info(
        "These are the previously reported Colab test results. "
        "Verify that they match the saved model and test split."
    )


elif page == "Live Prediction":
    st.subheader("Live Customer Churn Prediction")
    st.write(
        "Enter customer details and click Predict. "
        "The app uses the saved trained pipeline."
    )

    # Keep your existing prediction form and its code here.

    st.subheader("Live Customer Churn Prediction")
    st.write("Enter customer details and click Predict. The app uses the saved trained pipeline.")

    # Input options mirror the 19 features used by the fitted pipeline.
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        with col2:
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        with col3:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )
            monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=1.0)
            total = st.number_input("Total Charges", min_value=0.0, value=840.0, step=10.0)

        submitted = st.form_submit_button("Predict Churn", type="primary")

    if submitted:
        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        # Enforce exact fitted feature order and names
        try:
            customer = customer[expected_features]
            prediction = int(model.predict(customer)[0])
            if prediction == 1:
                st.error("Prediction: LIKELY TO CHURN")
            else:
                st.success("Prediction: LIKELY TO STAY")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(customer)[0]
                classes = list(getattr(model, "classes_", []))
                if 1 in classes:
                    churn_probability = float(probabilities[classes.index(1)])
                    st.metric("Estimated Churn Probability", f"{churn_probability:.1%}")
                    st.progress(min(max(churn_probability, 0.0), 1.0))
            st.caption("This is a model estimate, not a guarantee.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.info(
                "Check that the exported model's expected feature names and input categories match this form."
            )

st.sidebar.divider()
st.sidebar.caption("Customer Churn Prediction | CA4")
