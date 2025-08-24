# src/training/train.py

import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Assuming data_processing.py is in the same directory or accessible
from data_processing import create_features 

# --- Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "etf_features.parquet"

MODEL_NAME = "etf-trend-predictor-lr"

# --- Main Training Function ---
def train_model():
    """
    Trains the champion model (Logistic Regression) and registers it with MLflow.
    """
    mlflow.set_experiment("ETF_Trend_Prediction")
    
    with mlflow.start_run(run_name="LogisticRegression_Champion_Training") as run:
        print("--- Starting training run ---")

        # 1. Load and process data
        print("Loading and processing data...")
        all_tickers = ['SPY', 'QQQ', 'UVXY', 'TLT', 'GLD', 'AAPL', 'MSFT']
        all_data = [pd.read_csv(RAW_DATA_DIR / f'{ticker}.csv', index_col='Date', parse_dates=True) for ticker in all_tickers]
        raw_df = pd.concat(all_data, axis=1).dropna()
        
        processed_data = create_features(raw_df)
        
        X = processed_data.drop('target', axis=1)
        y = processed_data['target']
        
        # 2. Split data chronologically
        split_date = '2022-01-01'
        X_train, X_test = X.loc[:split_date], X.loc[split_date:]
        y_train, y_test = y.loc[:split_date], y.loc[split_date:]
        
        # 3. Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 4. Train model
        print("Training Logistic Regression model...")
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # 5. Evaluate model
        y_pred = model.predict(X_test_scaled)
        f1 = f1_score(y_test, y_pred)
        print(f"Test F1 Score: {f1:.4f}")
        
        # 6. Log to MLflow
        print("Logging to MLflow...")
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("test_f1_score", f1)
        
        # Log the scaler and model together in a pipeline for easy serving
        from sklearn.pipeline import make_pipeline
        pipeline = make_pipeline(scaler, model)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )
        
        # 7. Promote model using Tags (more robust than stages)
        print("Promoting model using tags...")
        client = mlflow.tracking.MlflowClient()

        # ... (the code to get new_model_version is the same) ...
        registered_model = client.get_registered_model(name=MODEL_NAME)
        latest_version_info = registered_model.latest_versions[0]
        new_model_version = latest_version_info.version

        print(f"Found newly created version: {new_model_version}. Setting 'status=staging' tag...")

        # --- THIS IS THE CHANGE ---
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=new_model_version,
            key="status",     # CHANGE: Use 'status' as the key
            value="staging"   # CHANGE: Use 'staging' as the value
        )

        # Optional but recommended cleanup loop
        for mv in registered_model.latest_versions:
            if mv.version != new_model_version and mv.tags.get("status") == "staging":
                print(f"Removing 'staging' tag from older version {mv.version}...")
                client.delete_model_version_tag(name=MODEL_NAME, version=mv.version, key="status")
                
        print(f"Successfully tagged model '{MODEL_NAME}' version {new_model_version} with 'status=staging'.")

if __name__ == '__main__':
    train_model()