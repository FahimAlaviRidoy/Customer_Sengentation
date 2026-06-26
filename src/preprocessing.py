import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from src.config import SAVED_MODELS_DIR

def preprocess_data(df):
    print("  Capping outliers...")
    for col in ['recency', 'frequency', 'monetary']:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = np.clip(df[col], lower, upper)
        
    print("  Applying log transformation...")
    for col in ['recency', 'frequency', 'monetary']:
        df[col] = np.log1p(df[col])
        
    print("  Scaling data...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['recency', 'frequency', 'monetary']])
    joblib.dump(scaler, f"{SAVED_MODELS_DIR}/scaler.pkl")
    return scaled_data, df
