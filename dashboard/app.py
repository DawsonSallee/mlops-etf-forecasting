# dashboard/app.py
import streamlit as st
import pandas as pd
import mlflow
import sys
from pathlib import Path

# --- Path Setup ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from config import MLFLOW_TRACKING_URI, MODEL_NAME

# --- Page Config ---
st.set_page_config(page_title="Model Comparison Dashboard", page_icon="🔬", layout="wide")

# --- MLflow Setup ---
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

# --- Data Loading ---
@st.cache_resource
def load_test_data_and_models():
    """Finds the most recent experiment run and loads its test data and all logged models."""
    # Fetch the most recent "champion" run to get the correct X_test
    try:
        champion_run = mlflow.search_runs(
            experiment_names=["ETF_Trend_Prediction"],
            filter_string="tags.mlflow.runName = 'XGBoost_Tuned_Champion'",
            order_by=["start_time DESC"],
            max_results=1
        ).iloc[0]
        run_id = champion_run.run_id
    except IndexError:
        return None, None, "Could not find a 'XGBoost_Tuned_Champion' run. Please execute the training pipeline."
    except Exception as e:
        return None, None, f"An error occurred fetching the latest run: {e}"

    # Download X_test artifact
    try:
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="shap_assets")
        x_test = pd.read_parquet(Path(local_path) / "X_test.parquet")
    except Exception as e:
        return None, None, f"Failed to load test data from run {run_id}: {e}"

    # --- Model Loading ---
    # Models are logged in separate runs, but we can find them in the same experiment
    models = {}
    model_run_names = {
        "Logistic Regression": "LogisticRegression_Baseline",
        "Random Forest": "RandomForest_Baseline",
        "XGBoost (Tuned)": "XGBoost_Tuned_Champion",
        "MLP": "MLP_Manual_Baseline"
    }
    model_artifact_paths = {
        "Logistic Regression": "logistic-regression-model",
        "Random Forest": "random-forest-model",
        "XGBoost (Tuned)": "xgb-model",
        "MLP": "mlp-model"
    }

    for name, run_name in model_run_names.items():
        try:
            run = mlflow.search_runs(
                experiment_names=["ETF_Trend_Prediction"],
                filter_string=f"tags.mlflow.runName = '{run_name}'",
                order_by=["start_time DESC"],
                max_results=1
            ).iloc[0]
            
            model_uri = f"runs:/{run.run_id}/{model_artifact_paths[name]}"
            models[name] = mlflow.pyfunc.load_model(model_uri)
        except (IndexError, Exception) as e:
            st.warning(f"Could not load model '{name}'. Run the full training notebook. Error: {e}")
            models[name] = None
            
    return x_test, models, None

# --- Main App ---
st.title("🔬 Model Comparison Dashboard")
st.markdown("Compare predictions from all trained models for the latest data point.")

x_test, models, error = load_test_data_and_models()

if error:
    st.error(error)
else:
    st.success("Test data and models loaded successfully!")
    
    st.subheader("Latest Data Point")
    latest_data = x_test.tail(1)
    st.dataframe(latest_data)

    st.subheader("Model Predictions")
    
    predictions = {}
    for name, model in models.items():
        if model:
            try:
                if name == "MLP":
                    # PyTorch model returns probabilities in a DataFrame
                    prob = model.predict(latest_data).iloc[0, 0]
                    pred_class = 1 if prob > 0.5 else 0
                else:
                    # Sklearn models return class labels in a numpy array
                    pred_class = model.predict(latest_data)[0]
                
                predictions[name] = "UP" if pred_class == 1 else "DOWN"
            except Exception as e:
                predictions[name] = f"Error: {e}"
        else:
            predictions[name] = "Not Loaded"
            
    # Display predictions in columns
    cols = st.columns(len(models))
    for idx, (name, pred) in enumerate(predictions.items()):
        with cols[idx]:
            st.metric(label=name, value=pred)

    st.info("Note: The MLP model requires scaled data for optimal performance. Predictions here are on unscaled data and may differ from its performance during training.")