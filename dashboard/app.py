# dashboard/app.py
import streamlit as st
import pandas as pd
import mlflow
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="ETF Trend Forecaster",
    page_icon="📈",
    layout="wide"
)

# --- Asset Loading ---
# This function connects to MLflow and loads the production model.
@st.cache_resource
def load_production_model_and_assets():
    # Set the tracking URI to the local mlruns directory
    # Streamlit Cloud runs from the repo root, so './mlruns' is the correct path
    mlflow.set_tracking_uri("file:./mlruns")
    
    model_name = "etf-xgboost-predictor"
    model_stage = "Production"
    
    # Load the production model as a pyfunc model
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")
    
    # To get the SHAP assets, we need the run_id of the model
    client = mlflow.tracking.MlflowClient()
    model_version_details = client.get_latest_versions(model_name, stages=[model_stage])[0]
    run_id = model_version_details.run_id
    
    # Download the SHAP explainer and X_test artifacts from the run
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id)
    explainer_path = Path(local_path) / "shap_explainer/explainer.joblib"
    xtest_path = Path(local_path) / "shap_xtest/X_test.parquet"
    
    with open(explainer_path, 'rb') as f:
        explainer = joblib.load(f)
        
    x_test_for_shap = pd.read_parquet(xtest_path)
    
    return model, explainer, x_test_for_shap

try:
    model, explainer, x_test_for_shap = load_production_model_and_assets()
except Exception as e:
    st.error(f"Error loading model from MLflow: {e}. Please ensure you've run the training and promotion scripts.")
    st.stop()
    
# --- The rest of the UI is the same! ---
st.title("📈 MLOps ETF Trend Forecaster")
st.write(
    "This application predicts the next-day price movement for the SPY ETF. "
    "The prediction is powered by the **production-stage** XGBoost model from our MLflow registry."
)

if st.button("Get Prediction for the Next Trading Day", type="primary"):
    with st.spinner("Analyzing latest market data..."):
        feature_vector = x_test_for_shap.tail(1)
        prediction_result = model.predict(feature_vector)[0]
        prediction_label = "UP" if prediction_result == 1 else "DOWN"
        
        # Note: predict_proba might not be available on all pyfunc models.
        # We can keep it simple for now.
        if prediction_label == "UP":
            st.success(f"The model predicts the market will go **UP**.")
        else:
            st.error(f"The model predicts the market will go **DOWN**.")

st.header("🧠 Model Explanation (SHAP)")
st.write("This chart shows what features influenced the prediction for the last data point in our test set.")

shap_values_instance = explainer.shap_values(x_test_for_shap.tail(1))
shap.initjs()
st.components.v1.html(shap.force_plot(explainer.expected_value, shap_values_instance, x_test_for_shap.tail(1)).data, height=150, scrolling=True)

st.write("### Overall Feature Importance")
summary_plot_path = Path(__file__).resolve().parent.parent / 'shap_summary_champion.png'
if summary_plot_path.exists():
    st.image(str(summary_plot_path))