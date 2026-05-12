import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from PIL import Image
import cv2

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
    try:
        df = pd.read_excel("./credit_scoring_20000.xlsx", sheet_name="Credit_Data")
        if len(df) > 12000:
            df = df.sample(8000, random_state=42).reset_index(drop=True)
        return df
    except Exception as e:
        st.error("❌ credit_scoring_20000.xlsx not found!")
        st.stop()

df = load_data()

# Feature Engineering
if "loan_to_income" not in df.columns:
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"].replace(0, 1)

# ====================== TRAIN / LOAD MODEL ======================
@st.cache_resource
def get_model():
    model_path = "crediguard_model.pkl"
    if os.path.exists(model_path):
        try:
            model, encoders, feature_cols = joblib.load(model_path)
            return model, encoders, feature_cols
        except:
            pass

    with st.spinner("Training model..."):
        data = df.copy()
        
        data['loan_to_income'] = data['loan_amount'] / data['annual_income']
        data['credit_utilization_high'] = (data['credit_utilization_pct'] > 50).astype(int)
        
        encoders = {}
        categorical_cols = ['home_ownership', 'purpose']
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le
        
        feature_cols = ['age', 'annual_income', 'employment_years', 'home_ownership',
                        'loan_amount', 'loan_term_months', 'interest_rate', 'purpose',
                        'num_late_payments', 'num_bankruptcies', 'credit_utilization_pct',
                        'loan_to_income']
        
        X = data[feature_cols]
        y = data['default']
        
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        
        joblib.dump((model, encoders, feature_cols), model_path)
        return model, encoders, feature_cols

model, encoders, feature_cols = get_model()

# ====================== SIDEBAR NAVIGATION ======================
page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Home", "📊 EDA Dashboard", "🔍 Single Applicant Assessment", 
     "📸 KYC Selfie Assessment"]
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
        fig = px.histogram(df, x="age", color="default", barmode="group", 
                          title="Age Distribution by Default Status")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig2 = px.box(df, x="home_ownership", y="annual_income", color="default", 
                     title="Income by Home Ownership")
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Risk Factors")
    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.scatter(df, x="loan_to_income", y="credit_utilization_pct", 
                         color="default", title="Loan-to-Income vs Credit Utilization")
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        fig4 = px.histogram(df, x="num_late_payments", color="default", 
                           title="Late Payments Impact")
        st.plotly_chart(fig4, use_container_width=True)

# ====================== SINGLE APPLICANT ASSESSMENT ======================
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
            'num_late_payments': num_late_payments,
            'num_bankruptcies': num_bankruptcies,
            'credit_utilization_pct': credit_util,
            'loan_to_income': loan_amount / annual_income
        }])
        
        for col, encoder in encoders.items():
            if col in input_data.columns:
                input_data[col] = encoder.transform(input_data[col].astype(str))
        
        for col in feature_cols:
            if col not in input_data.columns:
                input_data[col] = 0
        
        prob_default = model.predict_proba(input_data[feature_cols])[0][1]
        credit_score = max(300, int(850 - (prob_default * 550)))
        
        decision = "✅ APPROVE" if prob_default < 0.15 else "⚠️ REVIEW" if prob_default < 0.35 else "❌ DECLINE"
        
        st.success(f"**Assessment Complete for {name}**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Credit Score", credit_score, 
                     delta="Good" if credit_score >= 670 else "Fair")
        with col2:
            st.metric("Default Probability", f"{prob_default*100:.1f}%")
        with col3:
            st.metric("Final Decision", decision)

# ====================== KYC SELFIE ASSESSMENT ======================
elif page == "📸 KYC Selfie Assessment":
    st.header("📸 KYC Selfie Verification & Credit Decision")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Your Selfie")
        method = st.radio("Input Method", ["Live Camera", "Upload Photo"])
        photo = None
        if method == "Live Camera":
            photo = st.camera_input("Take clear selfie")
        else:
            uploaded = st.file_uploader("Upload Selfie", type=["jpg", "jpeg", "png"])
            if uploaded:
                photo = uploaded
        
        if photo:
            image = Image.open(photo)
            st.image(image, caption="Captured Selfie", use_column_width=True)
    
    with col2:
        st.subheader("🔍 Verification Result")
        if photo:
            # Face Detection
            opencv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                st.error("❌ No face detected.")
            else:
                st.success("✅ Face detected")
                
                # Randomized scoring for demo
                base_score = 720
                variation = np.random.randint(-35, 45)
                final_score = max(300, min(850, base_score + variation))
                
                prob_default = round(max(0.01, (850 - final_score) / 680), 3)
                
                if final_score >= 730:
                    dec, color = "✅ APPROVED", "green"
                    limit_options = [800000, 1000000, 1200000, 1500000, 1800000]
                elif final_score >= 650:
                    dec, color = "⚠️ REVIEW", "orange"
                    limit_options = [300000, 450000, 600000, 750000]
                else:
                    dec, color = "❌ DECLINED", "red"
                    limit_options = [0]
                
                recommended_limit = np.random.choice(limit_options)
                limit_str = f"₹{recommended_limit:,}" if recommended_limit > 0 else "Not Eligible"
                
                st.markdown(f"### Final Decision: <span style='color:{color}; font-size:1.5em'>{dec}</span>", 
                           unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                c1.metric("Base Score", base_score)
                c1.metric("Final Score", final_score)
                c2.metric("Default Prob", f"{prob_default*100:.1f}%")
                c2.metric("Recommended Limit", limit_str)
                
                if st.button("✅ Approve KYC"):
                    st.balloons()
                    st.success("KYC Approved Successfully!")

# ====================== SIDEBAR FOOTER ======================
st.sidebar.markdown("---")
st.sidebar.info("🛡️ CrediGuard AI v3.5\nCredit Scoring + KYC Selfie Engine")