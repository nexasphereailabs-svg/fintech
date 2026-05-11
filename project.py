
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from datetime import datetime

# ====================== CONFIG ======================
st.set_page_config(
    page_title="CrediGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ CrediGuard AI")
st.markdown("**Know Your Customer Before You Lend** — Intelligent Credit Scoring & Risk Engine")

# ====================== LOAD DATASET ======================
@st.cache_data
def load_data():
    df = pd.read_excel("./credit_scoring_20000.xlsx", sheet_name="Credit_Data")
    return df

df = load_data()

# Ensure engineered features exist for EDA
if "loan_to_income" not in df.columns:
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"].replace(0, np.nan)
    df["loan_to_income"] = df["loan_to_income"].fillna(0)


# ====================== TRAIN / LOAD MODEL ======================
@st.cache_resource
def train_model():
    if os.path.exists("crediguard_model.pkl"):
        model, encoders = joblib.load("crediguard_model.pkl")
        return model, encoders
    
    data = df.copy()
    
    # Feature Engineering
    data['loan_to_income'] = data['loan_amount'] / data['annual_income']
    data['credit_utilization_high'] = (data['credit_utilization_pct'] > 50).astype(int)
    
    # Encode categorical features
    encoders = {}
    categorical_cols = ['home_ownership', 'purpose']
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le
    
    # Create features used by both training + EDA.
    
    X = data[feature_cols]
    y = data['default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    joblib.dump((model, encoders), "crediguard_model.pkl")
    return model, encoders

model, encoders = train_model()

# ====================== SIDEBAR ======================
page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Home", "📊 EDA Dashboard", "🔍 Single Applicant Assessment", "📈 Batch Scoring", "📋 Portfolio Reports"]
)

# ====================== HOME ======================
if page == "🏠 Home":
    st.header("Welcome to CrediGuard AI")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Applicants", f"{len(df):,}")
    with col2:
        default_rate = (df['default'].mean() * 100)
        st.metric("Default Rate", f"{default_rate:.1f}%")
    with col3:
        st.metric("Avg Annual Income", f"${df['annual_income'].mean():,.0f}")
    with col4:
        st.metric("Avg Loan Amount", f"${df['loan_amount'].mean():,.0f}")

# ====================== EDA DASHBOARD ======================
elif page == "📊 EDA Dashboard":
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="age", color="default", barmode="group", title="Age Distribution by Default Status")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig2 = px.box(df, x="home_ownership", y="annual_income", color="default", title="Income by Home Ownership")
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Risk Factors")
    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.scatter(df, x="loan_to_income", y="credit_utilization_pct", color="default",
                         title="Loan-to-Income vs Credit Utilization")
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        fig4 = px.histogram(df, x="num_late_payments", color="default", title="Late Payments Impact")
        st.plotly_chart(fig4, use_container_width=True)

# ====================== SINGLE APPLICANT ======================
elif page == "🔍 Single Applicant Assessment":
    st.header("Single Applicant KYC & Credit Assessment")
    
    with st.form("applicant_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name", "John Doe")
            age = st.number_input("Age", 18, 80, 35)
            annual_income = st.number_input("Annual Income ($)", 10000, 500000, 65000)
            employment_years = st.number_input("Employment Years", 0, 50, 6)
            home_ownership = st.selectbox("Home Ownership", df['home_ownership'].unique())
        
        with col2:
            loan_amount = st.number_input("Loan Amount ($)", 1000, 500000, 25000)
            loan_term = st.number_input("Loan Term (months)", 12, 360, 36)
            interest_rate = st.number_input("Interest Rate (%)", 5.0, 30.0, 12.5)
            purpose = st.selectbox("Loan Purpose", df['purpose'].unique())
            num_late_payments = st.number_input("Number of Late Payments", 0, 20, 1)
            num_bankruptcies = st.number_input("Number of Bankruptcies", 0, 10, 0)
            credit_util = st.number_input("Credit Utilization (%)", 0.0, 100.0, 35.0)
        
        submitted = st.form_submit_button("Assess Risk & Generate Score")
    
    if submitted:
        input_data = pd.DataFrame([{
            'age': age,
            'annual_income': annual_income,
            'employment_years': employment_years,
            'home_ownership': home_ownership,
            'loan_amount': loan_amount,
            'loan_term_months': loan_term,
            'interest_rate': interest_rate,
            'purpose': purpose,
            'num_credit_lines': 10,
            'credit_utilization_pct': credit_util,
            'num_late_payments': num_late_payments,
            'num_bankruptcies': num_bankruptcies,
            'loan_to_income': loan_amount / annual_income
        }])
        
        # Encode categoricals
        for col, encoder in encoders.items():
            if col in input_data.columns:
                input_data[col] = encoder.transform(input_data[col].astype(str))
        
        prob_default = model.predict_proba(input_data)[0][1]
        credit_score = max(300, int(850 - (prob_default * 550)))
        
        decision = "✅ APPROVE" if prob_default < 0.15 else "⚠️ REVIEW" if prob_default < 0.35 else "❌ DECLINE"
        
        st.success(f"**Assessment Complete for {name}**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Credit Score", credit_score, delta="Good" if credit_score >= 670 else "Fair")
        with col2:
            st.metric("Default Probability", f"{prob_default*100:.1f}%")
        with col3:
            st.metric("Final Decision", decision)

# ====================== BATCH SCORING ======================
elif page == "📈 Batch Scoring":
    st.header("Batch Credit Scoring")
    uploaded = st.file_uploader("Upload new applications CSV (same format)", type=["csv"])
    
    if uploaded:
        batch = pd.read_csv(uploaded)
        st.write("Preview:", batch.head())
        
        if st.button("Score Batch"):
            # Add loan_to_income
            batch['loan_to_income'] = batch['loan_amount'] / batch['annual_income']
            
            for col, encoder in encoders.items():
                if col in batch.columns:
                    batch[col] = encoder.transform(batch[col].astype(str))
            
            probs = model.predict_proba(batch)[:, 1]
            scores = np.maximum(300, (850 - probs * 550).astype(int))
            decisions = np.where(probs < 0.15, "APPROVED", np.where(probs < 0.35, "REVIEW", "DECLINED"))
            
            batch['credit_score'] = scores
            batch['default_probability'] = probs
            batch['decision'] = decisions
            
            st.dataframe(batch, use_container_width=True)
            st.download_button("Download Scored Results", batch.to_csv(index=False), "scored_applications.csv")

# ====================== REPORTS ======================
elif page == "📋 Portfolio Reports":
    st.header("Portfolio Insights & Reports")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Approval Rate", f"{(1 - df['default'].mean())*100:.1f}%")
    col2.metric("High Risk Applicants", f"{(df['default'].sum()):,}")
    col3.metric("Avg Credit Utilization", f"{df['credit_utilization_pct'].mean():.1f}%")
    
    if st.button("Generate Full Portfolio Report"):
        st.success("📄 Professional Report Generated!")
        st.download_button("Download Excel Report", df.to_excel("crediguard_portfolio_report.xlsx", index=False), "crediguard_portfolio_report.xlsx")

st.sidebar.markdown("---")
st.sidebar.info("🛡️ CrediGuard AI\nBuilt for intelligent lending decisions")