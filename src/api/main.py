# src/api/main.py

import mlflow
import yfinance as yf
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager # Import the new context manager
from pathlib import Path
import sys
import logging

# --- 1. Setup & Configuration ---

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

try:
    from training.data_processing import create_features
except ImportError:
    logging.error("Could not import 'create_features'. Make sure 'src/training/data_processing.py' exists.")
    sys.exit(1)

# --- 2. Model Loading & Lifespan Management ---

# Create a dictionary to hold our model and its metadata
# This will be populated during the startup event
ml_models = {}

MODEL_NAME = "etf-trend-predictor-lr"
MODEL_TAG_KEY = "status"
MODEL_TAG_VALUE = "staging"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs ON STARTUP
    logging.info(f"App startup: Loading model '{MODEL_NAME}' with tag '{MODEL_TAG_KEY}={MODEL_TAG_VALUE}'")
    try:
        client = mlflow.tracking.MlflowClient()
        all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        
        staging_model_info = None
        for v in all_versions:
            if v.tags.get(MODEL_TAG_KEY) == MODEL_TAG_VALUE:
                staging_model_info = v
                break
        
        if staging_model_info:
            model_uri = f"models:/{MODEL_NAME}/{staging_model_info.version}"
            ml_models["predict_trend_model"] = mlflow.pyfunc.load_model(model_uri=model_uri)
            ml_models["metadata"] = {
                "name": MODEL_NAME,
                "version": staging_model_info.version,
                "tags": staging_model_info.tags,
                "run_id": staging_model_info.run_id
            }
            logging.info(f"Successfully loaded model version {ml_models['metadata']['version']}")
        else:
            logging.error(f"FATAL: No model version found with the tag '{MODEL_TAG_KEY}={MODEL_TAG_VALUE}'.")

    except Exception as e:
        logging.error(f"FATAL: An unexpected error occurred while loading the model: {e}", exc_info=True)
    
    yield
    
    # This code runs ON SHUTDOWN (after the 'yield')
    logging.info("App shutdown: Clearing ML models.")
    ml_models.clear()

# Pass the lifespan manager to the FastAPI app
app = FastAPI(
    title="ETF Trend Prediction API",
    description="An API to predict the next-day price trend of the SPY ETF.",
    version="1.0.0",
    lifespan=lifespan # This is the new, modern way
)


# --- 3. API Endpoints ---

@app.get("/")
def read_root():
    """ Root endpoint providing basic API and model information. """
    return {
        "message": "Welcome to the ETF Trend Prediction API",
        "model_info": ml_models.get("metadata", {}),
        "model_loaded_successfully": "predict_trend_model" in ml_models
    }

@app.post("/predict", tags=["Prediction"])
def predict_trend():
    """
    Fetches the latest market data, engineers features, and returns the 
    next-day trend prediction for the SPY ETF.
    """
    if "predict_trend_model" not in ml_models:
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to load. Please check server logs.")
    
    model = ml_models["predict_trend_model"]
    model_metadata = ml_models["metadata"]
    
    logging.info("Prediction request received. Fetching live market data...")
    
    try:
        # (The rest of your prediction logic remains exactly the same)
        tickers = ['SPY', 'QQQ', 'UVXY', 'TLT', 'GLD', 'AAPL', 'MSFT']
        raw_data = yf.download(tickers, period="250d", progress=False, auto_adjust=True)
        features_df = create_features(raw_data['Close'])
        
        if features_df.empty:
            logging.warning("Feature engineering resulted in an empty DataFrame.")
            raise HTTPException(status_code=500, detail="Could not generate features from the latest market data.")
            
        latest_features = features_df.iloc[[-1]]
        prediction_result = model.predict(latest_features)
        prediction_int = int(prediction_result[0])
        prediction_label = "UP" if prediction_int == 1 else "DOWN"
        
        logging.info(f"Prediction successful. Predicted trend: {prediction_label}")
        
        return {
            "prediction_date": (latest_features.index[0] + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
            "predicted_trend": prediction_label,
            "model_run_id": model_metadata.get("run_id", "unknown")
        }

    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")