import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.model import load_models
from src.features import rfm_features
from xgboost import plot_importance

st.set_page_config(page_title="CLV Prediction Dashboard", layout="wide")
st.title("🧠 Customer Lifetime Value (CLV) Prediction")
st.markdown("Compare Random Forest and XGBoost models with full insights & visualizations.")

# Sidebar Color Pickers
st.sidebar.header("🎨 Customize Graph Colors")
bar_color = st.sidebar.color_picker("Top 10 Customers Bar", "#008080")
scatter_color = st.sidebar.color_picker("Model Comparison Scatter", "#FF7F0E")
hist_color = st.sidebar.color_picker("RFM Distributions", "#2CA02C")
rf_importance_color = st.sidebar.color_picker("Random Forest Feature Importance", "#D62728")

uploaded_file = st.file_uploader("📁 Upload Customer Transactions CSV")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['InvoiceDate'], encoding='ISO-8859-1')

    df.rename(columns={
        'Invoice': 'InvoiceNo',
        'Price': 'UnitPrice',
        'Customer ID': 'CustomerID'
    }, inplace=True)

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate'])
    df = df[df['Quantity'] > 0]

    snapshot = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = rfm_features(df, snapshot)
    X = rfm[['Recency', 'Frequency', 'Monetary']]

    # Load models
    rf_model, xgb_model = load_models()

    # Predict CLV
    rfm['CLV_RF'] = rf_model.predict(X)
    rfm['CLV_XGB'] = xgb_model.predict(X)
    rfm = rfm.reset_index().rename(columns={'CustomerID': 'CustomerID'})

    st.subheader("📋 All CLV Predictions")
    st.dataframe(rfm)

    # ✅ Top 10 CLV Customers (XGBoost)
    st.subheader("🏆 Top 10 Customers by XGBoost CLV")
    top_10 = rfm.sort_values("CLV_XGB", ascending=False).head(10)

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(top_10['CustomerID'].astype(str), top_10['CLV_XGB'], color=bar_color)
    ax1.set_title("Top 10 Customers by XGBoost CLV")
    ax1.set_ylabel("Predicted CLV")
    ax1.set_xlabel("Customer ID")
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

    # ✅ Model Comparison: RF vs XGBoost
    st.subheader("📈 Model Comparison: Random Forest vs XGBoost")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=rfm['CLV_RF'], y=rfm['CLV_XGB'], ax=ax2, color=scatter_color)
    ax2.set_xlabel("Random Forest CLV")
    ax2.set_ylabel("XGBoost CLV")
    ax2.set_title("CLV Comparison")
    st.pyplot(fig2)

    # ✅ RFM Distributions
    st.subheader("📊 RFM Feature Distributions")
    fig3, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, feature in enumerate(['Recency', 'Frequency', 'Monetary']):
        sns.histplot(rfm[feature], ax=axes[i], kde=True, color=hist_color)
        axes[i].set_title(f'{feature} Distribution')
    st.pyplot(fig3)

    # ✅ Correlation Heatmap
    st.subheader("🧩 Correlation Matrix (RFM + CLV)")
    fig4, ax4 = plt.subplots(figsize=(6, 5))
    corr = rfm[['Recency', 'Frequency', 'Monetary', 'CLV_RF', 'CLV_XGB']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax4)
    st.pyplot(fig4)

    # ✅ Feature Importance (RF)
    st.subheader("🔍 Feature Importance: Random Forest")
    importances = rf_model.feature_importances_
    features = ['Recency', 'Frequency', 'Monetary']
    fig5, ax5 = plt.subplots()
    sns.barplot(x=importances, y=features, ax=ax5, color=rf_importance_color)
    ax5.set_title("RF Feature Importance")
    st.pyplot(fig5)

    # ✅ Feature Importance (XGBoost)
    st.subheader("🔍 Feature Importance: XGBoost")
    fig6, ax6 = plt.subplots(figsize=(8, 4))
    plot_importance(xgb_model, ax=ax6)
    st.pyplot(fig6)

    # ✅ Export predictions
    csv = rfm.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions CSV", csv, "CLV_predictions.csv", "text/csv")
