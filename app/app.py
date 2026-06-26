import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

st.title("Customer Segmentation Predictor")
st.write("Enter customer details to predict their segment.")

recency = st.number_input("Recency (Days since last purchase)", min_value=0, value=30)
frequency = st.number_input("Frequency (Number of transactions)", min_value=1, value=5)
monetary = st.number_input("Monetary (Total amount spent)", min_value=0.0, value=500.0)

model_choice = st.selectbox("Select Model", ("K-Means", "Agglomerative"))

if st.button("Predict Segment"):
    try:
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        if model_choice == "K-Means":
            model = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
        else:
            model = joblib.load(os.path.join(MODELS_DIR, 'agglomerative_model.pkl'))
            
         # Apply same preprocessing as training (log1p)
        # Use a Pandas DataFrame so feature names match the training data
        X_input = pd.DataFrame([[np.log1p(recency), np.log1p(frequency), np.log1p(monetary)]], columns=['recency', 'frequency', 'monetary'])
        X_scaled = scaler.transform(X_input)
        cluster = model.predict(X_scaled)[0]
        
        st.success(f"Predicted Cluster: **{cluster}**")
        
        if cluster == 0:
            st.info("Segment: **Champions**")
        elif cluster == 1:
            st.info("Segment: **Loyal Customers**")
        elif cluster == 2:
            st.info("Segment: **At Risk**")
        else:
            st.info("Segment: **Hibernating / Lost**")
            
    except FileNotFoundError:
        st.error("Models not found! Please run the training pipeline first.")
