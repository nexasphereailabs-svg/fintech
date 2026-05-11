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
import requests
from io import BytesIO

# ====================== CONFIG ======================
st.set_page_config(page_title="CrediGuard AI", page_icon="🛡️", layout="wide")
st.title("🛡️ CrediGuard AI")
st.markdown("**Know Your Customer Before You Lend** — Intelligent Credit Scoring & Risk Engine")

# ====================== LOAD DATA & MODEL ======================
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("./credit_scoring_20000.xlsx", sheet_name="Credit_Data")
        if len(df) > 10000:
            df = df.sample(8000, random_state=42).reset_index(drop=True)
        return df
    except:
        st.error("credit_scoring_20000.xlsx not found!")
        st.stop()

df = load_data()
if "loan_to_income" not in df.columns:
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"].replace(0, 1)

@st.cache_resource
def get_model():
    model_path = "crediguard_model.pkl"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except:
            pass

    with st.spinner("Training model..."):
        data = df.copy()
        data['loan_to_income'] = data['loan_amount'] / data['annual_income']
        encoders = {}
        for col in ['home_ownership', 'purpose']:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le
        
        feature_cols = ['age', 'annual_income', 'employment_years', 'home_ownership', 
                        'loan_amount', 'loan_term_months', 'purpose', 'num_late_payments',
                        'num_bankruptcies', 'credit_utilization_pct', 'loan_to_income']
        
        X = data[feature_cols]
        y = data['default']
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X, y)
        joblib.dump((model, encoders, feature_cols), model_path)
        return model, encoders, feature_cols

model, encoders, feature_cols = get_model()

# ====================== LOAD RECORDS ======================
@st.cache_data
def load_records():
    file_path = "records.csv"
    if not os.path.exists(file_path):
        st.error("❌ records.csv file not found!")
        st.stop()
    try:
        rec = pd.read_csv(file_path)
        if rec.empty:
            st.error("❌ records.csv is empty!")
            st.stop()
        st.sidebar.success(f"✅ Loaded {len(rec)} customers")
        return rec
    except Exception as e:
        st.error(f"Error reading records.csv: {e}")
        st.stop()

records = load_records()

# ====================== NAVIGATION ======================
page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Home", "📊 EDA Dashboard", "🔍 Single Applicant", 
     "📸 KYC Selfie Assessment", "📈 Batch Scoring", "📋 Reports"]
)

# ====================== HOME ======================
if page == "🏠 Home":
    st.header("Welcome to CrediGuard AI")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(records))
    col2.metric("Total Applications", len(df))
    if 'credit_score' in records.columns:
        col3.metric("Avg Credit Score", f"{records['credit_score'].mean():.0f}")

# ====================== KYC SELFIE ASSESSMENT ======================
if page == "📸 KYC Selfie Assessment":
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

                cust = records.iloc[0]

                # Show Registered Image
                if "image_url" in records.columns and pd.notna(cust.get("image_url")):
                    try:
                        resp = requests.get(cust["image_url"])
                        reg_img = Image.open(BytesIO(resp.content))
                        st.image(reg_img, caption="Registered Image", use_column_width=True)
                    except:
                        st.warning("Could not load registered image")

                # ==================== RANDOMIZE EVERY TIME ====================
                base_score = int(cust.get("credit_score", 700))
                variation = np.random.randint(-30, 40)
                final_score = max(300, min(850, base_score + variation))

                prob_default = round(max(0.01, (850 - final_score) / 680), 3)

                # Randomize Recommended Limit
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

                st.markdown(f"### Final Decision: <span style='color:{color}; font-size:1.5em'>{dec}</span>", unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                c1.metric("Base Score", base_score)
                c1.metric("Final Score", final_score)
                c2.metric("Default Prob", f"{prob_default*100:.1f}%")
                c2.metric("Recommended Limit", limit_str)

                if st.button("✅ Approve KYC"):
                    st.balloons()
                    st.success(f"KYC Approved Successfully!")

st.sidebar.markdown("---")
st.sidebar.info("CrediGuard AI v3.5\nRandomized Scores + Limits")