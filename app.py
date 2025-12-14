import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt

# ======================================================
# SAFE IMAGE LOADER (NO CRASH)
# ======================================================
def safe_image(path, width=420):
    if os.path.exists(path):
        st.image(path, width=width)
    else:
        st.info("Customer image not found (optional visual).")

# ======================================================
# CHART FUNCTIONS
# ======================================================
def plot_churn_vs_tenure():
    fig, ax = plt.subplots(figsize=(3, 2.5))
    ax.bar(["0–12", "12–24", "24+"], [65, 42, 18],
           color=["#E74C3C", "#F1C40F", "#2ECC71"])
    ax.set_title("Churn vs Tenure")
    ax.set_ylabel("Churn %")
    return fig

def plot_contract_churn():
    fig, ax = plt.subplots(figsize=(3, 2.5))
    ax.bar(["Month-to-Month", "One Year", "Two Year"],
           [43, 18, 7],
           color=["#E67E22", "#3498DB", "#2ECC71"])
    ax.set_title("Contract Impact")
    ax.set_ylabel("Churn %")
    return fig

def plot_charges_churn():
    fig, ax = plt.subplots(figsize=(3, 2.5))
    ax.bar(["Low", "Medium", "High"],
           [18, 27, 41],
           color=["#2ECC71", "#F1C40F", "#E74C3C"])
    ax.set_title("Pricing Effect")
    ax.set_ylabel("Churn %")
    return fig

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ======================================================
# STYLES (UNCHANGED CONTENT)
# ======================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
    color: #FDFEFE;
}
.kpi-card {
    text-align:center;
    padding:20px;
}
.kpi-icon { font-size:26px; }
.kpi-value { font-size:34px; font-weight:800; }
.kpi-label { color:#CDE6FF; }

.tile {
    background: rgba(255,255,255,0.08);
    padding:18px;
    border-radius:14px;
    margin-bottom:16px;
}

.float-tile {
    background: rgba(255,255,255,0.12);
    border-radius:22px;
    padding:26px;
    margin-bottom:22px;
}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL
# ======================================================
model = pickle.load(open("best_rf_model.pkl", "rb"))

# ======================================================
# SESSION STATE
# ======================================================
if "history" not in st.session_state:
    st.session_state.history = []

# ======================================================
# ANIMATED KPI COUNTER
# ======================================================
def animated_counter(label, value, icon):
    placeholder = st.empty()
    for i in range(0, value + 1, max(1, value // 30)):
        placeholder.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value">{i}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(0.02)

# ======================================================
# NAVIGATION
# ======================================================
page = st.radio(
    "Navigate",
    ["🏠 Home", "🔮 Predict", "📁 Batch Upload", "📊 Insights", "🕒 History", "ℹ️ About"],
    horizontal=True
)
st.divider()

# ======================================================
# HOME
# ======================================================
if page == "🏠 Home":

    st.markdown("## Customer Churn Prediction")
    st.markdown(
        "**Predict customer churn early using a production-ready Random Forest model**"
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1: animated_counter("Customers", 7043, "👥")
    with k2: animated_counter("ROC-AUC", 85, "📈")
    with k3: animated_counter("Threshold", 40, "🎯")
    with k4: animated_counter("Churn %", 27, "⚠️")

    st.markdown("""
    <div class="tile">
    💡 <b>Business Impact:</b> Enables early retention actions for high-risk customers,
    reducing revenue loss and improving customer lifetime value.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("### Key Drivers of Churn")
        c1, c2, c3 = st.columns(3)
        with c1: st.pyplot(plot_churn_vs_tenure())
        with c2: st.pyplot(plot_contract_churn())
        with c3: st.pyplot(plot_charges_churn())

    with right:
        safe_image("customers.png", width=420)

# ======================================================
# PREDICT
# ======================================================
elif page == "🔮 Predict":

    st.markdown("## Predict Customer Churn")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        with c2:
            monthly = st.slider("Monthly Charges", 0.0, 200.0, 70.0)
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        with c3:
            total = st.slider("Total Charges", 0.0, 10000.0, 2000.0)
            payment = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"]
            )
        submit = st.form_submit_button("Predict")

    if submit:
        df = pd.DataFrame({
            "tenure": [tenure],
            "MonthlyCharges": [monthly],
            "TotalCharges": [total],
            "Contract_Month-to-month": [int(contract == "Month-to-month")],
            "Contract_One year": [int(contract == "One year")],
            "InternetService_Fiber optic": [int(internet == "Fiber optic")],
            "InternetService_No": [int(internet == "No")],
            "PaymentMethod_Electronic check": [int(payment == "Electronic check")]
        })

        for col in model.feature_names_in_:
            if col not in df:
                df[col] = 0
        df = df[model.feature_names_in_]

        prob = model.predict_proba(df)[0][1]
        risk = "High" if prob >= 0.6 else "Medium" if prob >= 0.35 else "Low"

        st.markdown("<div class='float-tile'>", unsafe_allow_html=True)
        st.metric("Churn Probability", f"{prob:.2%}")
        st.progress(int(prob * 100))
        st.write(f"**Risk Category:** {risk}")
        st.markdown("</div>", unsafe_allow_html=True)

        report = f"""
Customer Churn Prediction Report
--------------------------------
Time: {datetime.now()}
Churn Probability: {prob:.2%}
Risk Category: {risk}
"""

        st.download_button(
            "📄 Download Prediction Report",
            report,
            file_name="churn_prediction_report.txt"
        )

        st.session_state.history.append({
            "Time": datetime.now(),
            "Probability": round(prob, 3),
            "Risk": risk
        })

# ======================================================
# BATCH UPLOAD
# ======================================================
elif page == "📁 Batch Upload":

    st.markdown("## Batch Churn Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        enc = pd.get_dummies(df)

        for col in model.feature_names_in_:
            if col not in enc:
                enc[col] = 0
        enc = enc[model.feature_names_in_]

        df["Churn_Probability"] = model.predict_proba(enc)[:, 1]
        st.dataframe(df.head())

        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            "batch_churn_predictions.csv"
        )

# ======================================================
# INSIGHTS
# ======================================================
elif page == "📊 Insights":

    st.markdown("## Key Business Insights")
    i1, i2, i3 = st.columns(3)
    with i1: st.pyplot(plot_churn_vs_tenure())
    with i2: st.pyplot(plot_contract_churn())
    with i3: st.pyplot(plot_charges_churn())

# ======================================================
# HISTORY
# ======================================================
elif page == "🕒 History":

    st.markdown("## Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.history))

# ======================================================
# ABOUT
# ======================================================
elif page == "ℹ️ About":

    st.markdown("## About This Project")
    st.markdown("""
    <div class="tile">
    <b>Customer Churn Prediction Using Random Forest</b> is a real-world machine learning
    project designed to help businesses proactively identify customers likely to churn.

    <br><br>
    <b>What this project demonstrates:</b><br>
    • Problem formulation and business understanding<br>
    • Exploratory Data Analysis<br>
    • Feature engineering and encoding<br>
    • Random Forest modeling and hyperparameter tuning<br>
    • Threshold optimization<br>
    • Model evaluation with ROC-AUC<br>
    • Deployment using Streamlit

    <br><br>
    <b>Tech Stack:</b> Python, Pandas, NumPy, Scikit-learn, Streamlit  
    <br><b>Author:</b> Indu Priya
    </div>
    """, unsafe_allow_html=True)

st.caption("Built with ❤️ by Indu Priya")
