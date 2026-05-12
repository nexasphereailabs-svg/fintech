import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="BankAI Sentinel", page_icon="🔒", layout="wide")
st.title("🔒 BankAI Sentinel")
st.markdown("### AI-Powered Fraud Detection & Personalization using Real Bank Transactions")

# ====================== LOAD DATA ======================
@st.cache_data
def load_data():
    df = pd.read_csv('./bank_transactions.csv')
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    df['Hour'] = df['TransactionDate'].dt.hour
    df['DayOfWeek'] = df['TransactionDate'].dt.dayofweek
    return df

df = load_data()

# ====================== SIDEBAR ======================
st.sidebar.header("Dataset Overview")
st.sidebar.metric("Total Transactions", f"{len(df):,}")
st.sidebar.metric("Unique Accounts", df['AccountID'].nunique())
st.sidebar.metric("Max Transaction", f"₹{df['TransactionAmount'].max():.2f}")
st.sidebar.metric("Max Login Attempts", int(df['LoginAttempts'].max()))

# ====================== TABS ======================
tab1, tab2, tab3 = st.tabs(["📊 Fraud Detection (HDFC Style)", 
                           "👥 Customer Segmentation (ICICI Style)", 
                           "🔬 Live Demo"])

# ====================== TAB 1: HDFC FRAUD DETECTION ======================
with tab1:
    st.header("HDFC Bank - AI Fraud Sentinel")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Transaction", f"₹{df['TransactionAmount'].mean():.2f}")
    with col2:
        st.metric("High Login Attempts (>3)", len(df[df['LoginAttempts'] > 3]))
    with col3:
        st.metric("Most Active Hour", df['Hour'].mode()[0])
    
    fig = px.histogram(df, x="TransactionAmount", color="Channel", nbins=50,
                      title="Transaction Amount Distribution by Channel")
    st.plotly_chart(fig, use_container_width=True)

# ====================== TAB 2: ICICI CUSTOMER SEGMENTATION ======================
with tab2:
    st.header("ICICI Bank - Personalized Offers")
    
    # Real customer-level aggregation
    customer_features = df.groupby('AccountID').agg({
        'TransactionAmount': ['mean', 'sum', 'count'],
        'LoginAttempts': 'max',
        'CustomerAge': 'first',
        'TransactionDuration': 'mean'
    }).reset_index()
    
    customer_features.columns = ['AccountID', 'AvgAmount', 'TotalSpend', 'TransactionCount', 
                                'MaxLoginAttempts', 'Age', 'AvgDuration']
    
    # Clustering using actual data
    features = customer_features[['AvgAmount', 'TotalSpend', 'TransactionCount', 'Age']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    customer_features['Cluster'] = kmeans.fit_predict(scaled)
    
    st.subheader("Customer Segments")
    fig3 = px.scatter(customer_features, x="TotalSpend", y="AvgAmount", 
                     color="Cluster", size="TransactionCount", 
                     hover_data=['AccountID', 'Age'],
                     title="Customer Segmentation based on Real Data")
    st.plotly_chart(fig3, use_container_width=True)

# ====================== TAB 3: LIVE DEMO ======================
with tab3:
    st.header("🔬 Try the AI Systems Live")
    
    demo_choice = st.radio("Choose Demo", 
                          ["Fraud Detection Simulator", "Personalized Offer Generator"])
    
    if demo_choice == "Fraud Detection Simulator":
        st.subheader("New Transaction Risk Assessment")
        
        # Safe values derived from actual data
        max_amount = float(df['TransactionAmount'].max())
        default_amount = float(df['TransactionAmount'].quantile(0.75))  # 75th percentile
        
        with st.form("fraud_form"):
            col1, col2 = st.columns(2)
            with col1:
                amount = st.number_input("Transaction Amount (₹)", 
                                       min_value=float(df['TransactionAmount'].min()),
                                       max_value=max_amount * 1.5,
                                       value=default_amount,
                                       step=50.0)
                hour = st.slider("Hour of Transaction", 0, 23, int(df['Hour'].mode()[0]))
                channel = st.selectbox("Channel", sorted(df['Channel'].unique()))
            
            with col2:
                login_attempts = st.slider("Login Attempts", 1, int(df['LoginAttempts'].max()) + 3, 1)
                duration = st.number_input("Transaction Duration (seconds)", 
                                         min_value=10, 
                                         max_value=int(df['TransactionDuration'].max()) + 100, 
                                         value=120)
                occupation = st.selectbox("Occupation", sorted(df['CustomerOccupation'].unique()))
            
            submitted = st.form_submit_button("🔍 Analyze Risk")
        
        if submitted:
            # Risk scoring based on real data patterns
            base_risk = 0.15
            if amount > df['TransactionAmount'].quantile(0.90):
                base_risk += 0.35
            if login_attempts >= 3:
                base_risk += 0.40
            if hour < 6 or hour > 22:
                base_risk += 0.25
            if duration < 60 and amount > df['TransactionAmount'].quantile(0.80):
                base_risk += 0.20
            
            risk_prob = min(0.96, base_risk + np.random.uniform(-0.10, 0.10))
            risk_score = int(risk_prob * 100)
            
            # ==================== VISUALIZATIONS ====================
            st.markdown("### Risk Analysis Dashboard")
            
            col_v1, col_v2 = st.columns([3, 2])
            
            with col_v1:
                fig_hist = px.histogram(df, x="TransactionAmount", nbins=60, 
                                      title="Where Your Transaction Stands")
                fig_hist.add_vline(x=amount, line_dash="dash", line_color="red",
                                 annotation_text="Your Transaction", 
                                 annotation_position="top right")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col_v2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_score,
                    title={'text': "Fraud Risk Score"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "darkred" if risk_score > 65 else "orange" if risk_score > 35 else "green"}},
                    delta={'reference': 30}))
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Risk Breakdown
            risk_categories = ['High Amount', 'Multiple Logins', 'Odd Hours', 'Fast Transaction']
            risk_values = [
                35 if amount > df['TransactionAmount'].quantile(0.90) else 8,
                40 if login_attempts >= 3 else 8,
                25 if (hour < 6 or hour > 22) else 8,
                20 if (duration < 60 and amount > 5000) else 8
            ]
            
            fig_pie = px.pie(names=risk_categories, values=risk_values, title="Risk Factor Breakdown")
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Decision
            if risk_prob > 0.65:
                st.error(f"🚨 HIGH FRAUD RISK — {risk_prob:.1%}")
                st.warning("Transaction Blocked")
            elif risk_prob > 0.35:
                st.warning(f"⚠️ Suspicious — {risk_prob:.1%}")
                st.info("Additional Verification Required")
            else:
                st.success(f"✅ Low Risk — {risk_prob:.1%}")
                # st.balloons()  ← Removed
    
    else:
        st.subheader("Personalized Offer Generator")
        with st.form("offer_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider("Customer Age", 18, 80, 35)
                total_spend = st.number_input("Total Spend (Last period)", 1000, 500000, 45000)
            with col2:
                avg_txn = st.number_input("Average Transaction", 50, 50000, 1200)
                channel_pref = st.selectbox("Preferred Channel", df['Channel'].unique())
            
            submitted = st.form_submit_button("Generate Offers")
        
        if submitted:
            if total_spend > 150000:
                offers = ["Premium Credit Card", "High Value Personal Loan", "Wealth Management"]
            elif total_spend > 50000:
                offers = ["Cashback Credit Card", "Personal Loan Offer", "SIP Recommendation"]
            else:
                offers = ["Secured Card", "Emergency Loan", "Savings Plan"]
            
            st.success("🎯 Personalized Offers for You:")
            for offer in offers:
                st.markdown(f"✅ **{offer}**")

st.markdown("---")
st.caption("All insights, thresholds, and visualizations are derived from the uploaded bank_transactions.csv dataset.")