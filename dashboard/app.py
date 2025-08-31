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
from config import MLFLOW_TRACKING_URI, MODEL_NAME, RAW_DATA_DIR, TARGET_TICKER

# --- Page Config ---
st.set_page_config(page_title="Model Comparison Dashboard", page_icon="🔬", layout="wide")

# --- MLflow Setup ---
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

# --- Data Loading ---
@st.cache_resource
def load_artifacts_and_models():
    """
    Finds the most recent experiment runs, loads test data, actual labels,
    and all logged models along with their metrics.
    """
    # Define model names and their corresponding run names and artifact paths
    model_configs = {
        "Logistic Regression": {"run_name": "LogisticRegression_Baseline", "artifact_path": "logistic-regression-model"},
        "Random Forest": {"run_name": "RandomForest_Baseline", "artifact_path": "random-forest-model"},
        "XGBoost (Tuned)": {"run_name": "XGBoost_Tuned_Champion", "artifact_path": "xgb-model"},
        "MLP": {"run_name": "MLP_Manual_Baseline", "artifact_path": "mlp-model"}
    }
    
    # Load data artifacts from the champion run first
    try:
        champion_run = mlflow.search_runs(
            experiment_names=["ETF_Trend_Prediction"],
            filter_string=f"tags.mlflow.runName = '{model_configs['XGBoost (Tuned)']['run_name']}'",
            order_by=["start_time DESC"], max_results=1
        ).iloc[0]
        run_id = champion_run.run_id
        
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="shap_assets")
        x_test = pd.read_parquet(Path(local_path) / "X_test.parquet")
        y_test = pd.read_parquet(Path(local_path) / "y_test.parquet")
    except (IndexError, Exception) as e:
        return None, None, None, None, f"Could not load champion run artifacts. Did you run the training pipeline? Error: {e}"

    # Load models and their metrics
    models = {}
    model_metrics = {}
    for name, config in model_configs.items():
        try:
            run = mlflow.search_runs(
                experiment_names=["ETF_Trend_Prediction"],
                filter_string=f"tags.mlflow.runName = '{config['run_name']}'",
                order_by=["start_time DESC"], max_results=1
            ).iloc[0]
            
            model_uri = f"runs:/{run.run_id}/{config['artifact_path']}"
            models[name] = mlflow.pyfunc.load_model(model_uri)
            
            # The MLP logs f1 score with a different key
            f1_metric_key = 'metrics.test_f1_score' if name == "MLP" else 'metrics.f1_score'
            
            model_metrics[name] = {
                "Accuracy": run.get('metrics.accuracy', None),
                "F1 Score": run.get(f1_metric_key, None),
                "ROC AUC": run.get('metrics.roc_auc', None)
            }
        except (IndexError, Exception):
            models[name] = None
            model_metrics[name] = {"Accuracy": "Not Found", "F1 Score": "Not Found", "ROC AUC": "Not Found"}
            
    return x_test, y_test, models, model_metrics, None

@st.cache_data
def load_spy_data():
    """Loads historical SPY data for plotting."""
    spy_path = RAW_DATA_DIR / f"{TARGET_TICKER}.csv"
    if spy_path.exists():
        df = pd.read_csv(spy_path, index_col='Date', parse_dates=True)
        return df[TARGET_TICKER]
    return None

def get_predictions(x_data, models):
    """Generates predictions for a given dataset from all models."""
    all_preds = {}
    for name, model in models.items():
        if model:
            if name == "MLP":
                probs = model.predict(x_data.copy())
                all_preds[name] = [1 if p > 0.5 else 0 for p in probs.iloc[:, 0]]
            else:
                all_preds[name] = model.predict(x_data.copy())
    return all_preds

def style_correctness(cell_value):
    """Styles the cell based on correctness."""
    if cell_value == "✅":
        return "color: green; font-weight: bold;"
    elif cell_value == "❌":
        return "color: red; font-weight: bold;"
    return ""

# --- Main App ---
st.markdown("<h1 style='text-align: center;'>🔮 SPY Next-Day Trend Forecast</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True) # Adds a little vertical space

x_test, y_test, models, model_metrics, error = load_artifacts_and_models()

if error:
    st.error(error)
else:
    # --- Latest Prediction ---
    latest_data = x_test.tail(1)
    latest_predictions = get_predictions(latest_data, models)
    
    cols = st.columns(len(models))
    for idx, (name, model) in enumerate(models.items()):
        if model:
            with cols[idx]:
                prediction = latest_predictions[name][0]
                if prediction == 1:
                    pred_label = "UP"
                    color = "green"
                else:
                    pred_label = "DOWN"
                    color = "red"
                
                # Centered and styled model name
                st.markdown(f"<h3 style='text-align: center; color: #a0a0a0; margin-bottom: 0;'>{name}</h3>", unsafe_allow_html=True)
                st.markdown(f'<p style="color:{color}; text-align: center; font-size: 2.5rem; font-weight: bold; margin-top: 0;">{pred_label}</p>', unsafe_allow_html=True)
        else:
             with cols[idx]:
                # Centered and styled model name
                st.markdown(f"<h3 style='text-align: center; color: #a0a0a0; margin-bottom: 0;'>{name}</h3>", unsafe_allow_html=True)
                st.markdown(f'<p style="text-align: center; font-size: 2.5rem; font-weight: bold; margin-top: 0;">Not Loaded</p>', unsafe_allow_html=True)

    # --- Recent Performance ---
    st.subheader("⚡️ Recent Performance (Last 5 Days)")
    last_5_x = x_test.tail(5)
    last_5_y = y_test.tail(5)
    
    predictions = get_predictions(last_5_x, models)
    
    history_df = pd.DataFrame(index=last_5_x.index)
    history_df["Actual Trend"] = last_5_y['target'].map({1: "UP", 0: "DOWN"})
    
    for name, preds in predictions.items():
        pred_labels = ["UP" if p == 1 else "DOWN" for p in preds]
        correctness = ["✅" if p == a else "❌" for p, a in zip(preds, last_5_y['target'])]
        history_df[f"{name} Prediction"] = pred_labels
        history_df[f"{name} Correct?"] = correctness
        
    column_order = ["Actual Trend"]
    for name in models:
        column_order.append(f"{name} Prediction")
        column_order.append(f"{name} Correct?")
    history_df = history_df[[col for col in column_order if col in history_df.columns]]
    
    styled_df = history_df.style.apply(lambda col: col.map(style_correctness), subset=[col for col in history_df.columns if "Correct?" in col])
    st.dataframe(styled_df, use_container_width=True)

    # --- Historical Price Chart ---
    spy_data = load_spy_data()
    if spy_data is not None:
        st.subheader(f"Historical {TARGET_TICKER} Closing Prices")
        st.line_chart(spy_data)