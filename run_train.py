import os
import json
import joblib
import pandas as pd
from src.database import BigQueryHandler
from src.engineering import engineer_features
from src.train import train_model

from dotenv import load_dotenv

load_dotenv()

def run_pipeline():
    # 1. Fetch Data
    print("--- Step 1: Fetching Data from BigQuery ---")
    
    # Load project_id from credentials.json
    with open("credentials.json", "r") as f:
        credentials = json.load(f)
    
    db = BigQueryHandler(project_id=credentials["project_id"]) 
    df = db.get_training_data()
    
    # 2. Engineering
    print("--- Step 2: Engineering Features ---")
    df_engineered = engineer_features(df)
    
    # Define features — drop metadata AND total_adds (used to define target → leakage)
    X = df_engineered.drop(columns=['user_pseudo_id', 'last_active', 'total_adds'], errors='ignore')

    # Real churn label: dataset-relative cutoff (handles historical/sample data correctly)
    # Using the most recent event in the dataset as "now" so we get a real class split
    df_engineered['last_active_ts'] = pd.to_datetime(df_engineered['last_active'], unit='us', errors='coerce')
    dataset_now = df_engineered['last_active_ts'].max()  # treat most recent user as "today"
    cutoff = dataset_now - pd.Timedelta(days=30)
    y = (df_engineered['last_active_ts'] < cutoff).astype(int)

    # Sanity check — print class distribution before training
    print(f"Churn label distribution: {y.value_counts().to_dict()} (0=active, 1=churned)")
    
    # 3. Train & Track
    print("--- Step 3: Training Model & Logging to W&B ---")
    model_pipeline = train_model(X, y)
    
    # 4. Save the Model
    print("--- Step 4: Saving Model for API ---")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model_pipeline, "models/churn_model.pkl")
    print("Successfully saved model to models/churn_model.pkl")

if __name__ == "__main__":
    run_pipeline()
