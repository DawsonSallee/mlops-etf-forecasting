# SPY Next-Day Trend Forecaster: An End-to-End MLOps Project

This project demonstrates a complete, end-to-end MLOps workflow for a time-series forecasting problem. It builds an automated system that acquires data, engineers features, trains multiple models, identifies a champion, and serves its predictions through a live, interactive Streamlit dashboard.

---

### Live Dashboard Preview

*(This is a placeholder for a GIF or screenshot of the final Streamlit application)*

![Dashboard Preview](https://via.placeholder.com/800x400.png?text=Live+Dashboard+Screenshot+Here)

---

### Executive Summary

The goal of this project is to move beyond a simple model in a Jupyter Notebook and build a robust, automated, and maintainable machine learning system. The system predicts the next-day price trend (UP or DOWN) for the SPY ETF by leveraging a rich set of engineered features from various market indicators.

This project showcases mastery in four key areas of modern data science and MLOps:
1.  **Data & Feature Engineering Pipeline**: A reproducible, automated pipeline for data acquisition and feature creation.
2.  **Advanced Model Training & Evaluation**: A rigorous, experiment-tracked process for developing a high-performance "champion" model.
3.  **MLOps Automation & Model Registry**: A "hands-off" workflow that trains, registers, and versions models, preparing them for production.
4.  **Interactive Serving & Explainability**: A user-friendly web application that serves live predictions from the best available model.

---

### System Architecture

The project follows a modular architecture that separates concerns, from data processing to model serving.

```text
[ Data Pipeline: Notebooks 01 -> 03 ]
             |
             v
[ ML Pipeline: Notebook 04 ] ------> [ MLflow Server ] <------ [ Streamlit Dashboard ]
(Trains & Logs Models)           (Stores Models & Metrics)    (Loads & Serves Models)
```

---

### Core Technical Pillars

#### 1. Data & Feature Engineering Pipeline

The foundation of any successful model is a robust and reliable data pipeline.

-   **Data Acquisition**: Historical daily price data for SPY and other correlated assets (QQQ, TLT, GLD, etc.) is programmatically downloaded using the `yfinance` library.
-   **Feature Engineering**: Over 30 predictive features are systematically engineered to capture different aspects of market dynamics. This includes:
    -   **Technical Indicators**: Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), Bollinger Bands, and Average True Range (ATR).
    -   **Lagged Returns & Volatility**: Historical returns and rolling volatility measures over multiple time windows (e.g., 5-day, 21-day) to capture momentum and risk.
    -   **Inter-Market Relationships**: Correlations and price ratios between SPY and other assets (e.g., SPY/GLD ratio) to model broader market sentiment.
-   **Data Storage**: The processed feature set is saved in the efficient Parquet format for fast read/write operations.

#### 2. Advanced Model Training & Evaluation

This project moves beyond a single model to a competitive evaluation framework, ensuring the best model is always selected.

-   **Baseline Models**: Simple yet powerful models like **Logistic Regression** and **Random Forest** are trained to establish a performance benchmark.
-   **Champion Model (XGBoost)**: A **Gradient Boosting Machine (XGBoost)** is trained as the primary "champion" model due to its high performance and interpretability in tabular data contexts.
    -   **Hyperparameter Tuning with Optuna**: The `Optuna` framework is used to perform automated, efficient hyperparameter optimization. It intelligently searches the parameter space to find the optimal combination for metrics like F1-score.
    -   **Time-Series Cross-Validation**: To prevent lookahead bias, a `TimeSeriesSplit` cross-validation strategy is employed during the tuning process, ensuring the model is always validated on "future" data relative to its training set.
-   **Challenger Model (MLP)**: A **Multi-Layer Perceptron (MLP)**, built with PyTorch, serves as a deep learning challenger model. This demonstrates the flexibility of the pipeline to incorporate different modeling frameworks.
-   **Experiment Tracking with MLflow**: Every single training run is meticulously logged in an **MLflow Tracking Server**. This includes:
    -   Model hyperparameters.
    -   Performance metrics (Accuracy, F1-Score, ROC AUC).
    -   The trained model object itself as a versioned artifact.

#### 3. MLOps Automation & Model Registry

The core of the MLOps workflow is the automated training pipeline (`04_model_training_and_evaluation.ipynb`), which acts as a single, executable unit.

-   **Atomic Training Runs**: When executed, the pipeline automatically trains all models, compares their performance, and logs all results to MLflow.
-   **MLflow Model Registry**: The best-performing model (the tuned XGBoost model) is registered in the MLflow Model Registry. This creates a versioned, centralized "model store" that separates the model artifact from the code that produced it.
-   **Automated Promotion to Production**: The script automatically promotes the newly registered model version to the "Production" stage. This simulates a real-world Continuous Training (CT) process where the best model is automatically deployed.

#### 4. Interactive Serving

The final output is a user-facing Streamlit dashboard that makes the model's predictions accessible and useful.

-   **Dynamic Model Loading**: On startup, the dashboard queries the MLflow Model Registry to find and load the model currently in the "Production" stage. This ensures the app is always serving predictions from the best available model without requiring a code change or redeployment.
-   **Live Predictions**: The dashboard presents the latest predictions from all trained models in a clear, color-coded UI.
-   **Interactive Data Visualization**: An interactive historical price chart, built with `Plotly`, allows users to explore SPY's price history with moving averages and a date range slider.

---

### How to Run This Project

Follow these steps to set up the environment and run the complete pipeline.

**1. Clone the Repository**

```bash
git clone <your-repo-url>
cd mlops-etf-forecasting
```

**2. Set Up the Python Environment**

It is highly recommended to use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
pip install -r requirements.txt
```

**3. Run the MLOps Pipeline**

Execute the notebooks in sequence to run the data pipeline and train the models. This will populate the `mlruns` directory with all the experiment data.

```bash
# Run the data acquisition and feature engineering pipelines
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_acquisition.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_feature_engineering.ipynb

# Run the model training, evaluation, and registration pipeline
jupyter nbconvert --to notebook --execute --inplace notebooks/04_model_training_and_evaluation.ipynb
```

**4. Launch the Streamlit Dashboard**

Once the pipeline has been run successfully, you can launch the interactive dashboard.

```bash
streamlit run dashboard/app.py
```

You should now be able to access the live dashboard in your web browser!