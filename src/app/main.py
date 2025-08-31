# src/app/main.py
from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd

# In a full refactor, these would be imported. We define them here for simplicity.
def download_latest_data():
    print("Simulating live data download...")
    try:
        df = pd.read_parquet("data/processed/etf_features.parquet")
        if df.shape[0] < 200:
             raise ValueError("Not enough data for feature calculation")
        return df.tail(200)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Processed data file not found for simulation.")

def generate_inference_features(live_data_df):
    print("Generating features for inference...")
    return live_data_df.tail(1).drop(columns=['target'], errors='ignore')

# --- FastAPI App ---
app = FastAPI(title="ETF Trend Forecaster API", version="1.0")

# Set the tracking URI for the container environment
mlflow.set_tracking_uri("file:/app/mlruns")

# Load the production model from the MLflow Model Registry on startup
MODEL_NAME = "etf-xgboost-predictor"
MODEL_STAGE = "Production"
model = None
try:
    client = mlflow.tracking.MlflowClient()
    # Get the latest version of the model in the "Production" stage
    prod_version = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])[0]
    
    # Construct a portable URI using the model's run_id and the artifact path
    # This forces MLflow to look for artifacts relative to the run, bypassing the bad absolute path.
    model_uri = f"runs:/{prod_version.run_id}/xgb-model"
    
    print(f"Loading model '{MODEL_NAME}' version {prod_version.version} from URI: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Successfully loaded model '{MODEL_NAME}'")
    
except Exception as e:
    print(f"FATAL: Could not load model from MLflow. Check server logs. Error: {e}")

@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome to the ETF Trend Forecaster API."}

@app.post("/predict", tags=["Prediction"])
def predict():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available.")
    try:
        latest_data = download_latest_data()
        feature_vector = generate_inference_features(latest_data)
        prediction_result = model.predict(feature_vector)[0]
        prediction_label = "UP" if prediction_result == 1 else "DOWN"
        
        return {
            "prediction": prediction_label,
            "model_name": MODEL_NAME,
            "model_stage": MODEL_STAGE
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
