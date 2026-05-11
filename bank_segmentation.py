# ICICI Bank Customer Segmentation Dashboard using Streamlit
# Run using:
# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="ICICI Bank Customer Segmentation",
    page_icon="🏦",
    layout="wide"
)

# -------------------------------
# TITLE
# -------------------------------
st.title("🏦 ICICI Bank - Customer Segmentation Dashboard")
st.markdown("""
This project demonstrates **Customer Segmentation using K-Means Clustering + PCA**
for behavioral analysis of banking customers.

### Objective
Identify customer groups based on:
- Income
- Spending Pattern
- Savings
- Loan Amount
- Credit Card Usage
- Digital Transactions

This helps ICICI Bank in:
- Personalized Offers
- Cross-selling Insurance & Loans
- Targeted Marketing
- Improving Customer Lifetime Value
""")

# -------------------------------
# GENERATE SAMPLE DATA
# -------------------------------
np.random.seed(42)

n_customers = 500

data = pd.DataFrame({
    "Income": np.random.randint(20000, 200000, n_customers),
    "Monthly_Spending": np.random.randint(5000, 100000, n_customers),
    "Savings": np.random.randint(10000, 500000, n_customers),
    "Loan_Amount": np.random.randint(0, 1000000, n_customers),
    "Credit_Card_Usage": np.random.randint(1000, 150000, n_customers),
    "Digital_Transactions": np.random.randint(1, 300, n_customers)
})

st.subheader("📄 Customer Dataset")
st.dataframe(data.head())

# -------------------------------
# DATA PREPROCESSING
# -------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# -------------------------------
# K-MEANS CLUSTERING
# -------------------------------
k = st.sidebar.slider("Select Number of Clusters", 2, 10, 4)

kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

data["Cluster"] = clusters

# -------------------------------
# PCA FOR VISUALIZATION
# -------------------------------
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)

data["PCA1"] = pca_components[:, 0]
data["PCA2"] = pca_components[:, 1]

# -------------------------------
# DISPLAY CLUSTERS
# -------------------------------
st.subheader("📊 Customer Segmentation Result")

fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(
    data["PCA1"],
    data["PCA2"],
    c=data["Cluster"],
    cmap="viridis",
    s=80
)

ax.set_title("Customer Segmentation using K-Means + PCA")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")

legend = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend)

st.pyplot(fig)

# -------------------------------
# CLUSTER INSIGHTS
# -------------------------------
st.subheader("📈 Cluster Insights")

cluster_summary = data.groupby("Cluster").mean(numeric_only=True)

st.dataframe(cluster_summary)

# -------------------------------
# BUSINESS INSIGHTS
# -------------------------------
st.subheader("💡 Business Insights")

for cluster in sorted(data["Cluster"].unique()):
    st.markdown(f"### Cluster {cluster}")

    avg_income = cluster_summary.loc[cluster, "Income"]
    avg_spending = cluster_summary.loc[cluster, "Monthly_Spending"]

    if avg_income > 120000:
        st.success(
            "High-income customers. Recommend premium credit cards, "
            "wealth management, and investment products."
        )

    elif avg_spending > 60000:
        st.info(
            "High spending customers. Good target for cashback offers, "
            "shopping rewards, and personal loans."
        )

    else:
        st.warning(
            "Regular banking customers. Recommend savings plans, "
            "insurance, and digital banking offers."
        )

# -------------------------------
# DOWNLOAD OPTION
# -------------------------------
csv = data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download Segmented Customer Data",
    data=csv,
    file_name="icici_customer_segmentation.csv",
    mime="text/csv"
)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Developed for ICICI Bank FinTech Customer Segmentation Project 🚀")