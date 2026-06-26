import mlflow
import mlflow.sklearn
import joblib
import os
from src.config import SAVED_MODELS_DIR
from src.data_loader import load_and_prepare_data
from src.preprocessing import preprocess_data
from src.eda import perform_eda
from src.models.kmeans_model import train_kmeans
from src.models.agglomerative_model import train_agglomerative

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Customer_Segmentation")
    
    with mlflow.start_run(run_name="full_pipeline"):
        print("1. Loading data...")
        df = load_and_prepare_data()
        
        print("2. Performing EDA...")
        perform_eda(df)
        
        print("3. Preprocessing...")
        X_scaled, df_processed = preprocess_data(df)
        
        print("4. Training K-Means...")
        km_model, km_labels = train_kmeans(X_scaled)
        mlflow.sklearn.log_model(km_model, "kmeans_model")
        joblib.dump(km_model, f"{SAVED_MODELS_DIR}/kmeans_model.pkl")
        
        print("5. Training Agglomerative...")
        agg_model, agg_labels = train_agglomerative(X_scaled)
        mlflow.sklearn.log_model(agg_model, "agglomerative_model")
        joblib.dump(agg_model, f"{SAVED_MODELS_DIR}/agglomerative_model.pkl")
        
        df_processed['kmeans_cluster'] = km_labels
        df_processed['agg_cluster'] = agg_labels
        df_processed.to_csv(f"{SAVED_MODELS_DIR}/segmented_customers.csv", index=False)
        
        print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
