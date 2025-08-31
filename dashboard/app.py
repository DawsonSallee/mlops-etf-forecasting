# dashboard/app.py
import streamlit as st
import pandas as pd
import mlflow
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(page_title="ETF Trend Forecaster", page_icon="📈", layout="wide")

# --- Asset Loading ---
@st.cache_resource
def load_production_model_and_assets():
    # This path construct works when running `streamlit run dashboard/app.py` from the root
    project_root = Path(__file__).resolve().parent.parent
    mlruns_path = "file:" + str(project_root / "mlruns")
    mlflow.set_tracking_uri(mlruns_path)
    
    model_name = "etf-xgboost-predictor"
    
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Production")
    
    client = mlflow.tracking.MlflowClient()
    model_version_details = client.get_latest_versions(model_name, stages=["Production"])[0]
    run_id = model_version_details.run_id
    
    # Download artifacts to a temporary directory
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="shap_assets")
    
    explainer = joblib.load(Path(local_path) / "explainer.joblib")
    x_test_for_shap = pd.read_parquet(Path(local_path) / "X_test.parquet")
    
    return model, explainer, x_test_for_shap

try:
    model, explainer, x_test_for_shap = load_production_model_and_assets()
    st.success("Production model and SHAP assets loaded successfully from MLflow!")
except Exception as e:
    st.error(f"Error loading model from MLflow: {e}. Please ensure you have run the full training pipeline first.")
    st.stop()
    
# --- UI Components ---
st.title("📈 ETF Trend Forecaster")
st.write("This application uses the **production-stage** XGBoost model from our MLflow registry to predict the next-day price movement for the SPY ETF.")

if st.button("Get Prediction", type="primary"):
    with st.spinner("Analyzing latest data..."):
        feature_vector = x_test_for_shap.tail(1)
        prediction = model.predict(feature_vector)[0]
        label = "UP" if prediction == 1 else "DOWN"
        
        if label == "UP": st.success(f"Prediction: **{label}**")
        else: st.error(f"Prediction: **{label}**")

        st.subheader("🧠 Prediction Explanation")
        shap_values = explainer.shap_values(feature_vector)
        st.pyplot(shap.force_plot(explainer.expected_value, shap_values, feature_vector, matplotlib=True))

st.subheader("Overall Feature Importance")
st.pyplot(shap.summary_plot(explainer.shap_values(x_test_for_shap), x_test_for_shap, plot_type='bar'))