# AI-Based Mobile Banking Fraud Detection System

An end-to-end, AI-powered fraud detection prototype for Bangladesh's mobile financial services (MFS) ecosystem. The project covers the full lifecycle: synthetic data generation, exploratory analysis, feature engineering, training and comparing multiple machine-learning models, SHAP-based explainability and risk scoring, a FastAPI prediction service, and an interactive Streamlit dashboard.

> **Note:** The system is built entirely on **synthetic data** for research/prototyping purposes. No real customer or transaction data is used.

## Features

- **Synthetic dataset** of 1,000,000 transactions across 100,000 customers, 5,000 agents, and 10,000 merchants, with a 5% fraud rate and 10 modeled fraud scenarios (account takeover, SIM swap, rapid transactions, money laundering, account mule, transaction splitting, suspicious cash-out, unusual location, new device, agent collusion).
- **Exploratory data analysis (EDA)** with automated charts and a data-quality report.
- **Feature engineering** with behavioral, temporal, risk, and agent/merchant features, plus a leakage-safe chronological train/validation/test split (70/15/15).
- **Multiple models** trained and compared: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, and an unsupervised Isolation Forest, with class balancing and F1-optimized decision thresholds.
- **Explainable AI** using SHAP for global and local explanations, plus a 0–100 risk score with recommended actions.
- **REST API** (FastAPI) for single and batch fraud predictions, analytics, and model performance.
- **Interactive dashboard** (Streamlit) with overview, transaction monitoring, fraud alerts, and model performance pages.

## Project Structure

```text
.
├── api/
│   └── main.py                     # FastAPI prediction & analytics service
├── configs/                        # Reserved for configuration files
├── dashboard/
│   └── app.py                      # Streamlit dashboard
├── data/
│   ├── raw/                        # Synthetic source tables (customers, agents, merchants, transactions)
│   └── processed/                  # Train / validation / test splits after feature engineering
├── models/                         # Trained model & preprocessing artifacts (*.pkl)
├── notebooks/                      # Reserved for analysis notebooks
├── reports/
│   ├── eda/                        # EDA charts and data-quality report
│   ├── model_results/              # Per-model feature-importance CSVs
│   ├── model_comparison.csv        # Model metrics comparison
│   └── results.json                # Full evaluation results
├── src/
│   ├── data/feature_pipeline.py    # Feature engineering and dataset splitting
│   ├── models/train.py             # Model training, evaluation and artifact export
│   └── explainability/shap_explainer.py  # SHAP analysis and risk scoring
├── tests/                          # Pytest test suites
├── eda_report.py                   # Standalone EDA script
├── requirements.txt
└── README.md
```

## Dataset

The raw data lives in `data/raw/` as both CSV and Parquet files:

| Table | Rows | Key columns |
| --- | ---: | --- |
| `transactions` | 1,000,000 | `transaction_id`, `timestamp`, `provider`, `transaction_type`, `amount_bdt`, sender/receiver IDs and locations, device/network attributes, `is_fraud`, `fraud_type` |
| `customers` | 100,000 | `customer_id`, `account_age_days`, `kyc_level`, home location, `customer_risk_score` |
| `agents` | 5,000 | `agent_id`, location, `agent_risk_score` |
| `merchants` | 10,000 | `merchant_id`, location, `merchant_category` |

The dataset covers the full year 2025, with transaction amounts ranging from 10 to ~180,000 BDT. The strongest fraud correlates in the EDA are `failed_login_attempts`, `amount_bdt`, `new_device`, `amount_deviation_prior`, `unusual_location`, and `sim_changed`.

The feature pipeline adds 22 engineered features (behavioral history, time-since-previous, device/location/SIM risk flags, velocity features, and agent/merchant activity counts), producing 35 numeric model features.

## Workflow

Run the phases in order from the project root:

```powershell
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Exploratory data analysis  ->  reports/eda/
python eda_report.py

# 3. Feature engineering  ->  data/processed/, models/preprocessing/
python -m src.data.feature_pipeline

# 4. Train and evaluate models  ->  models/*.pkl, reports/
python -m src.models.train

# 5. SHAP explainability and risk-scoring demo (prints to console)
python -m src.explainability.shap_explainer
```

## Serving

### REST API

```powershell
python -m api.main
# Equivalent: uvicorn api.main:app --reload
```

The API runs on `http://localhost:8000`; interactive docs are at `/docs` and `/redoc`.

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | API information |
| `GET` | `/health` | Service health and loaded model |
| `POST` | `/api/v1/fraud/predict` | Score a single transaction |
| `POST` | `/api/v1/fraud/batch-predict` | Score a batch of up to 10,000 transactions |
| `GET` | `/api/v1/transactions` | Sample transactions |
| `GET` | `/api/v1/fraud/alerts` | Aggregate fraud alert statistics |
| `GET` | `/api/v1/analytics/overview` | Volume, fraud rate, and distributions |
| `GET` | `/api/v1/models/performance` | Best model and evaluation metrics |

Responses include the fraud probability, a 0–100 risk score (`Low`, `Medium`, `High`, `Critical`), a recommended action, and top contributing factors.

### Streamlit dashboard

```powershell
streamlit run dashboard/app.py
```

The dashboard provides a dataset-split selector and four pages: **Overview**, **Transaction Monitoring**, **Fraud Alerts**, and **Model Performance**.

## Model Results

Decision thresholds are tuned on the validation split to maximize F1. Test-set results from the committed run:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM (best) | 0.9908 | 0.9511 | 0.9705 | 0.9986 | 0.9869 |
| XGBoost | – | – | 0.9701 | – | – |
| Decision Tree | – | – | 0.9692 | – | – |
| Random Forest | – | – | 0.9608 | – | – |
| Logistic Regression | – | – | 0.9573 | – | – |
| Isolation Forest | – | – | 0.6519 | – | – |

LightGBM's most important features are `amount_bdt`, `failed_login_attempts`, `sender_amount_deviation_prior`, and `sender_amount_mean_prior`. Full metrics are in `reports/results.json` and `reports/model_comparison.csv`.

## Testing

```powershell
python -m pytest tests/ -v
```

The tests validate dataset size and quality, schema, fraud distribution, feature engineering, split sizes, risk scoring, and the presence of trained model artifacts. Note: the current tests use a hardcoded absolute project path and may need adjustment if the repository is moved.

## Tech Stack

- **Data & ML:** NumPy, pandas, SciPy, PyArrow, scikit-learn, XGBoost, LightGBM, imbalanced-learn
- **Explainability:** SHAP
- **Visualization:** Matplotlib, seaborn, Plotly
- **API:** FastAPI, Uvicorn, Pydantic
- **Dashboard:** Streamlit
- **Persistence:** SQLAlchemy, psycopg2-binary (declared for future use)
- **Testing:** pytest, httpx
- **Utilities:** joblib, tqdm, python-dotenv, PyYAML

## License

MIT License — Copyright (c) 2026 Monower Hossen. See [LICENSE](LICENSE).
