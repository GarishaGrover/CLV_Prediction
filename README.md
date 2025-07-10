# 🧠 Customer Lifetime Value (CLV) Prediction

This project predicts **Customer Lifetime Value** using **RFM (Recency, Frequency, Monetary)** features and compares two machine learning models: **Random Forest** and **XGBoost**. It also includes an interactive **Streamlit dashboard** with visualization and export features.

---

## 📊 Key Features

- Upload your own transaction dataset (CSV format)
- Automated **RFM feature engineering**
- CLV prediction using:
  - ✅ Random Forest Regressor
  - ✅ XGBoost Regressor
- Full comparison of predicted CLVs
- Visual insights:
  - Top 10 customers by CLV
  - Model performance comparison
  - RFM feature distributions
  - Correlation matrix
  - Feature importance (RF & XGB)
- Interactive dashboard with custom color options
- Download predictions as CSV

---

## 📁 Project Structure
CLV_Prediction/
├── app/
│ └── main.py                                         # Streamlit Dashboard
├── src/
│ ├── preprocessing.py                                # Data cleaning
│ ├── features.py                                     # RFM feature engineering
│ └── model.py                                        # Model training/loading
├── data/
│ └── raw/
│ └── online_retail.csv                               # Sample input file (replaceable)
├── models/
│ └── rf_model.pkl                                    # Saved Random Forest model
│ └── xgb_model.pkl                                   # Saved XGBoost model
├── train_model.py                                    # Script to train and save models
├── requirements.txt                                  # Required libraries

---

## 🗂️ Input CSV Format

Your dataset should contain the following columns:

| Column         | Description                            |
|----------------|----------------------------------------|
| Invoice        | Invoice number                         |
| StockCode      | Product code                           |
| Description    | Product name                           |
| Quantity       | Number of items bought                 |
| InvoiceDate    | Transaction date (in datetime format)  |
| Price          | Unit price of the product              |
| Customer ID    | Unique customer identifier             |
| Country        | Customer's country                     |

Make sure `InvoiceDate` is in proper date format. Recommended encoding: `ISO-8859-1`.

---

## 🚀 How to Run

### 1. Clone the Repository
git clone https://github.com/yourusername/clv-prediction.git
cd clv-prediction

2. Install Requirements
pip install -r requirements.txt

4. Train Models
python train_model.py

This generates:
rf_model.pkl – Random Forest model
xgb_model.pkl – XGBoost model

4. Launch Streamlit Dashboard
streamlit run app/main.py

6. Upload CSV and Visualize
Once the app opens, upload your dataset and explore all insights.

---

🧪 Algorithms Used
Model	                                     Strengths
Random Forest	                             Fast, interpretable, low variance
XGBoost	                                   Robust, high accuracy, handles outliers

---

📈 Visualizations
Top 10 Customers by CLV (bar chart)
RFM Feature Distributions (histograms)
Model Comparison (scatter plot)
Correlation Matrix (heatmap)
Feature Importances (bar charts for RF & XGB)
All charts support custom color options.

---

📤 Export Predictions
Download complete predictions in .csv format with columns:
CustomerID
Recency, Frequency, Monetary
CLV_RF – predicted by Random Forest
CLV_XGB – predicted by XGBoost

---

🧾 License
This project is licensed under the MIT License.
Use it freely for academic, personal, or business purposes.

---

🙋 Author
Garisha Grover
📧 garishagrover0803@gmail.com
🔗 GitHub: github.com/GarishaGrover
