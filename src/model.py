import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def train_models(X, y):
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    joblib.dump(rf, "models/rf_model.pkl")

    # XGBoost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb.fit(X, y)
    joblib.dump(xgb, "models/xgb_model.pkl")

def load_models():
    rf = joblib.load("models/rf_model.pkl")
    xgb = joblib.load("models/xgb_model.pkl")
    return rf, xgb
