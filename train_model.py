
from src.preprocessing import load_data
from src.features import rfm_features
from src.model import train_models
import pandas as pd

# Load your dataset (adjust path as needed)
df = pd.read_csv(r"C:\Users\Garisha Grover\Celebal\CLV_Prediction_Template\data\raw\online_retail.csv", parse_dates=['InvoiceDate'], encoding='ISO-8859-1')

# Preprocess
df.rename(columns={
    'Invoice': 'InvoiceNo',
    'Price': 'UnitPrice',
    'Customer ID': 'CustomerID'
}, inplace=True)

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df.dropna(subset=['InvoiceDate'], inplace=True)
df = df[df['Quantity'] > 0]

snapshot = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Feature engineering
rfm = rfm_features(df, snapshot)
X = rfm[['Recency', 'Frequency', 'Monetary']]
y = rfm['Monetary']  # You can replace with actual CLV if available

# Train both models
train_models(X, y)
print("✅ Models trained and saved!")
