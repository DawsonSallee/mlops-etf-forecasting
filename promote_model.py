# promote_model.py
import mlflow
from mlflow.tracking import MlflowClient

# --- Configuration ---
MLFLOW_TRACKING_URI = "./mlruns"  # Points to your local mlruns directory
EXPERIMENT_NAME = "ETF_Trend_Prediction"
MODEL_NAME = "etf-xgboost-predictor"

def promote_best_model():
    """
    Finds the best run in an experiment based on a metric,
    registers it, and promotes it to the "Production" stage.
    """
    print("--- Starting Model Promotion ---")
    
    # Initialize the MLflow client
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # 1. Get the experiment by name
    try:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")
        experiment_id = experiment.experiment_id
        print(f"Found experiment '{EXPERIMENT_NAME}' (ID: {experiment_id})")
    except Exception as e:
        print(f"Error getting experiment: {e}")
        return

    # 2. Find the best run within that experiment
    # We search for the run with the highest 'f1_score' metric
    # and filter for only the XGBoost runs.
    best_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="run_name = 'XGBoost_Tuned_Champion'",
        order_by=["metrics.f1_score DESC"],
        max_results=1
    )

    if not best_runs:
        print("Error: No 'XGBoost_Tuned_Champion' runs found in the experiment.")
        return

    best_run = best_runs[0]
    best_run_id = best_run.info.run_id
    best_f1_score = best_run.data.metrics["f1_score"]
    print(f"Found best run with ID: {best_run_id}")
    print(f"  -> F1 Score: {best_f1_score:.4f}")

    # 3. Register the model from the best run's artifacts
    # The artifact path 'xgb-model' is what you used in your training notebook.
    model_uri = f"runs:/{best_run_id}/xgb-model"
    print(f"Registering model from URI: {model_uri}")
    
    try:
        model_version = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        print(f"Successfully registered model '{MODEL_NAME}', version {model_version.version}")
    except Exception as e:
        print(f"Error registering model: {e}")
        # It's possible the model is already registered. Let's try to find it.
        latest_versions = client.get_latest_versions(MODEL_NAME, stages=["None", "Staging", "Production"])
        for v in latest_versions:
            if v.run_id == best_run_id:
                model_version = v
                print(f"Model was already registered. Found version {model_version.version}.")
                break
        if not model_version:
            print("Could not register or find a matching registered model.")
            return

    # 4. Transition the new model version to the "Production" stage
    print(f"Transitioning version {model_version.version} of model '{MODEL_NAME}' to 'Production'...")
    try:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True # This moves any old "Production" model to "Archived"
        )
        print("  -> Success!")
    except Exception as e:
        print(f"Error transitioning model stage: {e}")
        return
        
    print("\n--- Model Promotion Complete ---")

if __name__ == "__main__":
    promote_best_model()