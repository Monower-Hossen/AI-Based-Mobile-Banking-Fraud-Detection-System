#!/usr/bin/env python
# FastAPI Backend for AI-MFS-Fraud-Detection

import os
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.train import split_features_target, prepare_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fraud-detection-api')

# Project paths
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Load models and artifacts
loaded_models = {}
scaler = None
feature_info = None

# Pydantic models for API requests/responds

class FraudPredictionRequest(BaseModel):
    """Request model for single transaction fraud prediction."""
    amount_bdt: float = Field(..., description="Transaction amount in BDT")
    hour: int = Field(..., ge=0, le=23, description="Transaction hour (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon..6=Sun)")
    device_type: str = Field(..., description="Device type (ANDROID, FEATURE_PHONE, IOS, WEB)")
    network_type: str = Field(..., description="Network type (MOBILE_NAT, WIFI, BROADBAND, OTHER)")
    sim_changed: int = Field(..., ge=0, le=1, description="SIM changed indicator")
    new_device: int = Field(..., ge=0, le=1, description="New device indicator")
    unusual_location: int = Field(..., ge=0, le=1, description="Unusual location indicator")
    failed_login_attempts: int = Field(..., ge=0, description="Failed login attempts count")
    transaction_type: str = Field(..., description="Transaction type")
    is_night: int = Field(..., ge=0, le=1, description="Is night indicator")
    agent_risk_score: float = Field(..., description="Agent risk score")
    merchant_category: str = Field(..., description="Merchant category")
    sender_tx_count_prior: int = Field(..., ge=0, description="Prior sender tx count")
    sender_amount_mean_prior: float = Field(..., description="Prior sender mean amount")
    sender_amount_deviation_prior: float = Field(..., description="Amount deviation from prior mean")
    sender_high_value_prior: int = Field(..., ge=0, le=1, description="High value indicator")
    is_weekend: int = Field(..., ge=0, le=1, description="Weekend indicator")
    is_night_tx: int = Field(..., ge=0, le=1, description="Night transaction indicator")
    device_risk_flag: int = Field(..., ge=0, le=1, description="Device risk flag")
    location_risk_flag: int = Field(..., ge=0, le=1, description="Location risk flag")
    sim_risk_flag: int = Field(..., ge=0, le=1, description="SIM risk flag")
    multi_risk_flag: int = Field(..., ge=0, le=1, description="Multi-risk flag")
    high_value_tx: int = Field(..., ge=0, le=1, description="High value transaction indicator")
    very_high_value_tx: int = Field(..., ge=0, le=1, description="Very high value transaction indicator")
    rapid_tx_flag: int = Field(..., ge=0, le=1, description="Rapid transaction indicator")
    sender_unique_receivers_prior: int = Field(..., ge=0, description="Unique receivers prior")
    sender_time_since_prev_tx: float = Field(..., ge=0, description="Time since previous tx (hours)")
    sender_tx_count_15min: int = Field(..., ge=0, description="Tx count in last 15 min")

    @validator('hour')
    def hour_must_be_valid(cls, v):
        if v < 0 or v > 23:
            raise ValueError('Hour must be between 0 and 23')
        return v

    @validator('day_of_week')
    def day_must_be_valid(cls, v):
        if v < 0 or v > 6:
            raise ValueError('Day of week must be between 0 and 6')
        return v


class FraudPredictionResponse(BaseModel):
    """Response model for fraud prediction."""
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of fraud (0-1)")
    risk_score: int = Field(..., ge=0, le=100, description="Risk score 0-100")
    risk_category: str = Field(..., description="Risk category (Low, Medium, High, Critical)")
    recommended_action: str = Field(..., description="Recommended review action")
    main_contributing_factors: List[str] = Field(..., description="Top contributing factors")
    prediction_timestamp: str = Field(..., description="ISO format prediction timestamp")


class BatchPredictionRequest(BaseModel):
    """Request model for batch fraud prediction."""
    transactions: List[FraudPredictionRequest] = Field(..., description="List of transactions to predict")

    @validator('transactions')
    def transactions_not_empty(cls, v):
        if not v:
            raise ValueError('At least one transaction required')
        if len(v) > 10000:
            raise ValueError('Maximum 10000 transactions per batch')
        return v


class TransactionSummary(BaseModel):
    """Summary model for transaction overview."""
    transaction_id: str
    amount_bdt: float
    hour: int
    day_of_week: int
    is_fraud: bool
    fraud_probability: float
    risk_score: int


class HealthCheck(BaseModel):
    """Health check response model."""
    status: str
    model_loaded: str
    version: str = "1.0.0"


# FastAPI app
app = FastAPI(
    title="AI-MFS-Fraud-Detection API",
    description="""
    API for detecting fraud in Mobile Financial Services (MFS) transactions.
    Focused on Bangladesh's mobile banking ecosystem.
    
    This is an academic/prototype system using synthetic data.
    """,
    version="1.0.0",
    docs_url="/docs",  # Swagger UI endpoint
    redoc_url="/redoc"
)


# Global initialization
def load_artifacts():
    """Load trained models and preprocessing artifacts."""
    global scaler, feature_info, loaded_models
    
    logger.info("Loading model artifacts...")
    
    # Load scaler (may live in models/ or models/preprocessing/)
    scaler_candidates = [
        os.path.join(MODEL_DIR, 'scaler.pkl'),
        os.path.join(MODEL_DIR, 'preprocessing', 'scaler.pkl'),
    ]
    scaler_path = next((p for p in scaler_candidates if os.path.exists(p)), None)
    if scaler_path:
        scaler = joblib.load(scaler_path)
        logger.info(f"Scaler loaded from {scaler_path}")
    else:
        logger.warning("Scaler not found, will use manual scaling")
        scaler = None

    # Load feature info (may live in models/ or models/preprocessing/)
    feature_info_candidates = [
        os.path.join(MODEL_DIR, 'preprocessing', 'feature_info.pkl'),
        os.path.join(MODEL_DIR, 'feature_info.pkl'),
    ]
    feature_info_path = next((p for p in feature_info_candidates if os.path.exists(p)), None)
    if feature_info_path:
        feature_info = joblib.load(feature_info_path)
        logger.info(f"Feature info loaded from {feature_info_path}")
    else:
        logger.warning("Feature info not found")
        feature_info = None
    
    # Load best model (LightGBM)
    model_names = [
        'lightgbm_model.pkl',
        'xgboost_model.pkl', 
        'random_forest_model.pkl',
        'decision_tree_model.pkl',
        'logistic_regression_model.pkl'
    ]
    
    for model_name in model_names:
        model_path = os.path.join(MODEL_DIR, model_name)
        if os.path.exists(model_path):
            loaded_models[model_name] = joblib.load(model_path)
            logger.info(f"Loaded model: {model_name}")
            # Use LightGBM as the default/model with best performance
            if model_name == 'lightgbm_model.pkl':
                break
    
    if not loaded_models:
        logger.error("No models found! API will not function properly.")
    
    logger.info(f"Artifacts loaded: {len(loaded_models)} models, scaler={'yes' if scaler else 'no'}")


# Initialize artifacts on startup
load_artifacts()


# Helper functions

def build_feature_matrix(data_dict: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Build a model-ready feature matrix in the exact column order the scaler/model expect.

    Features present in the request are used as-is; features the model needs but
    the request does not carry are imputed with the scaler's training mean
    (which becomes 0 after standardization, i.e. a neutral value).
    Returns None when the scaler is unavailable.
    """
    if scaler is None:
        return None

    feature_names = getattr(scaler, 'feature_names_in_', None)
    cols = list(feature_names) if feature_names is not None else []
    if not cols and feature_info:
        cols = list(feature_info.get('numeric_cols', []))
    if not cols:
        return None

    means = getattr(scaler, 'mean_', None)
    row: Dict[str, float] = {}
    for i, col in enumerate(cols):
        raw = data_dict.get(col, None)
        try:
            val = float(raw)
            if np.isnan(val):
                raise ValueError
        except (TypeError, ValueError):
            val = float(means[i]) if means is not None and i < len(means) else 0.0
        row[col] = val

    X = pd.DataFrame([row], columns=cols)
    try:
        X_scaled = scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=cols)
    except Exception as exc:
        logger.warning(f"Feature scaling failed, using raw features: {exc}")
        return X


def prepare_single_input(data_dict: Dict[str, Any]) -> tuple:
    """Prepare a single transaction for model prediction."""
    # Create DataFrame from input
    df = pd.DataFrame([data_dict])
    
    # Get numeric features - exclude identifier columns
    exclude = ['transaction_id', 'sender_id', 'receiver_id', 'is_fraud', 'fraud_type']
    numeric_cols = [c for c in df.columns if c not in exclude + [c for c in df.columns if df[c].dtype == object]]
    
    # Select only numeric columns that exist
    available_numeric = [c for c in numeric_cols if c in df.columns]
    
    if not available_numeric:
        # Fallback: use all columns we can
        available_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    
    X = df[available_numeric].copy()
    
    # Fill NaN
    for c in available_numeric:
        if X[c].isnull().any():
            X[c] = X[c].fillna(X[c].median())
    
    # Scale if scaler available
    if scaler is not None:
        numeric_cols_for_scaling = [c for c in available_numeric if scaler.feature_names_in_ is not None and c in scaler.feature_names_in_]
        if numeric_cols_for_scaling:
            X[numeric_cols_for_scaling] = scaler.transform(X[numeric_cols_for_scaling])
    
    return X, df


def get_risk_score(fraud_probability: float) -> Dict[str, Any]:
    """Convert fraud probability to risk score and category."""
    low_thresh = 0.3
    high_thresh = 0.7
    critical_thresh = 0.9
    
    prob = float(fraud_probability)
    
    if prob >= critical_thresh:
        risk_score = int(100 * (prob - critical_thresh) / (1.0 - critical_thresh))
        risk_score = min(max(risk_score, 80), 100)
        risk_category = "Critical Risk"
        review_action = "Immediate review required - flag for manual inspection"
    elif prob >= high_thresh:
        risk_score = int(100 * (prob - high_thresh) / (critical_thresh - high_thresh)) + 60
        risk_score = min(max(risk_score, 60), 79)
        risk_category = "High Risk"
        review_action = "Priority review - monitor closely"
    elif prob >= low_thresh:
        risk_score = int(100 * (prob - low_thresh) / (high_thresh - low_thresh)) + 30
        risk_score = min(max(risk_score, 30), 59)
        risk_category = "Medium Risk"
        review_action = "Standard review - verify transaction"
    else:
        risk_score = int(100 * prob / low_thresh)
        risk_score = min(max(risk_score, 0), 29)
        risk_category = "Low Risk"
        review_action = "No review needed - transaction is routine"
    
    return {
        'fraud_probability': prob,
        'risk_score': risk_score,
        'risk_category': risk_category,
        'review_action': review_action
    }


def get_contributing_factors(model, X_row: pd.DataFrame, feature_names: List[str]) -> List[str]:
    """Get top contributing factors for a prediction."""
    try:
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # Fallback: use absolute SHAP-like values based on feature values
            importances = np.abs(X_row.select_dtypes(include=[np.number]).values[0])
            importances = importances[:len(feature_names)] if len(importances) >= len(feature_names) else \
                        np.pad(importances, (0, max(0, len(feature_names) - len(importances))))
        
        # Get top indices
        n_top = min(5, len(feature_names))
        top_indices = importances.argsort()[-n_top:][::-1]
        
        factors = []
        for idx in top_indices:
            if idx < len(feature_names):
                factors.append(feature_names[idx])
        
        return factors
    except Exception:
        return ["amount_bdt", "new_device", "unusual_location"]


# API Endpoints

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    model_name = "none"
    if loaded_models:
        model_name = list(loaded_models.keys())[0]
    
    return HealthCheck(
        status="ok" if loaded_models else "degraded",
        model_loaded=model_name
    )


@app.post("/api/v1/fraud/predict", response_model=FraudPredictionResponse)
async def predict_fraud(request: FraudPredictionRequest):
    """Predict fraud probability for a single transaction."""
    try:
        # Prepare input data
        data_dict = request.dict()

        # Build a correctly ordered, scaled feature matrix
        X = build_feature_matrix(data_dict)

        # Get fraud probability from model
        probability = 0.5  # default when the model is unavailable
        if X is not None and loaded_models:
            model = loaded_models.get('lightgbm_model.pkl') or next(iter(loaded_models.values()))
            try:
                prob = model.predict_proba(X)[:, 1][0]
                probability = float(np.clip(prob, 0, 1))
            except Exception as exc:
                logger.warning(f"Model prediction failed: {exc}")
                probability = 0.5

        # Get risk score and category
        risk_info = get_risk_score(probability)

        # Get contributing factors
        contributing_factors = get_contributing_factors(
            loaded_models.get('lightgbm_model.pkl'),
            X if X is not None else pd.DataFrame(),
            list(X.columns) if X is not None else [],
        )
        
        # Format response
        response = FraudPredictionResponse(
            fraud_probability=probability,
            risk_score=risk_info['risk_score'],
            risk_category=risk_info['risk_category'],
            recommended_action=risk_info['review_action'],
            main_contributing_factors=contributing_factors,
            prediction_timestamp=pd.Timestamp.now().isoformat()
        )
        
        logger.info(f"Prediction: prob={probability:.4f}, risk={risk_info['risk_category']}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/fraud/batch-predict", response_model=List[FraudPredictionResponse])
async def batch_predict_fraud(request: BatchPredictionRequest):
    """Predict fraud probability for multiple transactions."""
    try:
        results = []
        
        for tx in request.transactions:
            data_dict = tx.dict()

            # Build a correctly ordered, scaled feature matrix
            X = build_feature_matrix(data_dict)

            # Get probability from model
            probability = 0.5
            if X is not None and loaded_models:
                model = loaded_models.get('lightgbm_model.pkl') or next(iter(loaded_models.values()))
                try:
                    prob = model.predict_proba(X)[:, 1][0]
                    probability = float(np.clip(prob, 0, 1))
                except Exception as exc:
                    logger.warning(f"Batch model prediction failed: {exc}")
                    probability = 0.5

            # Get risk info
            risk_info = get_risk_score(probability)

            # Get contributing factors
            contributing_factors = get_contributing_factors(
                loaded_models.get('lightgbm_model.pkl'),
                X if X is not None else pd.DataFrame(),
                list(X.columns) if X is not None else [],
            )
            
            response = FraudPredictionResponse(
                fraud_probability=probability,
                risk_score=risk_info['risk_score'],
                risk_category=risk_info['risk_category'],
                recommended_action=risk_info['review_action'],
                main_contributing_factors=contributing_factors,
                prediction_timestamp=pd.Timestamp.now().isoformat()
            )
            results.append(response)
        
        logger.info(f"Batch prediction: {len(request.transactions)} transactions")
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/transactions", response_model=List[TransactionSummary])
async def get_transactions():
    """Get transaction summaries."""
    try:
        # Load sample transactions from processed data
        train_path = os.path.join(PROCESSED_DIR, 'train.parquet')
        if os.path.exists(train_path):
            train_df = pd.read_parquet(train_path)
            # Take a small sample for summary
            sample = train_df.head(50)
            
            summaries = []
            for _, row in sample.iterrows():
                # Determine if fraud based on is_fraud column
                is_fraud = bool(row.get('is_fraud', 0))
                fraud_prob = row.get('is_fraud', 0)  # use is_fraud as proxy
                
                # Get risk score
                risk_info = get_risk_score(fraud_prob)
                
                summaries.append(TransactionSummary(
                    transaction_id=row.get('transaction_id', 'unknown'),
                    amount_bdt=float(row.get('amount_bdt', 0)),
                    hour=int(row.get('hour', 0)),
                    day_of_week=int(row.get('day_of_week', 0)),
                    is_fraud=is_fraud,
                    fraud_probability=float(fraud_prob),
                    risk_score=risk_info['risk_score']
                ))
            return summaries
        
        return []
    except Exception as e:
        logger.error(f"Error getting transactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/fraud/alerts")
async def get_fraud_alerts():
    """Get fraud alerts from the system."""
    try:
        # Load training data to identify high-risk transactions
        train_path = os.path.join(PROCESSED_DIR, 'train.parquet')
        if os.path.exists(train_path):
            train_df = pd.read_parquet(train_path)
            
            # Find high-risk transactions (high is_fraud or high predicted risk)
            fraud_count = int(train_df['is_fraud'].sum())
            total_count = len(train_df)
            fraud_rate = fraud_count / total_count
            
            # Get some alert statistics
            alerts = {
                "total_transactions": total_count,
                "flagged_fraud": fraud_count,
                "fraud_rate": round(fraud_rate, 4),
                "critical_risk_count": 0,  # placeholder
                "high_risk_count": 0,  # placeholder
                "message": "Alert data loaded from training dataset"
            }
            return alerts
        
        return {"message": "No data available"}
    except Exception as e:
        logger.error(f"Error getting fraud alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analytics/overview")
async def get_analytics_overview():
    """Get analytics overview."""
    try:
        train_path = os.path.join(PROCESSED_DIR, 'train.parquet')
        if os.path.exists(train_path):
            train_df = pd.read_parquet(train_path)
            
            # Basic statistics
            total_transactions = len(train_df)
            fraud_count = int(train_df['is_fraud'].sum())
            fraud_rate = fraud_count / total_transactions
            total_volume = float(train_df['amount_bdt'].sum())
            avg_amount = float(train_df['amount_bdt'].mean())
            
            # Transaction type distribution
            tx_type_dist = train_df['transaction_type'].value_counts().to_dict() if 'transaction_type' in train_df.columns else {}
            
            # Hour distribution
            hour_dist = train_df['hour'].value_counts().to_dict() if 'hour' in train_df.columns else {}
            
            overview = {
                "total_transactions": total_transactions,
                "fraud_rate": round(fraud_rate, 4),
                "total_transaction_volume_bdt": round(total_volume, 2),
                "average_transaction_amount_bdt": round(avg_amount, 2),
                "transaction_type_distribution": tx_type_dist,
                "hour_distribution": {str(k): v for k, v in sorted(hour_dist.items())},
                "risk_distribution": {
                    "low": 0,  # placeholder
                    "medium": 0,
                    "high": 0,
                    "critical": 0
                }
            }
            return overview
        
        return {"message": "No data available"}
    except Exception as e:
        logger.error(f"Error getting analytics overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models/performance")
async def get_model_performance():
    """Get model performance metrics."""
    try:
        comparison_candidates = [
            os.path.join(PROJECT_ROOT, 'reports', 'model_comparison.csv'),
            os.path.join(PROJECT_ROOT, 'reports', 'model_results', 'model_comparison.csv'),
        ]
        results_candidates = [
            os.path.join(PROJECT_ROOT, 'reports', 'results.json'),
            os.path.join(PROJECT_ROOT, 'reports', 'model_results', 'results.json'),
        ]
        comparison_path = next((p for p in comparison_candidates if os.path.exists(p)), comparison_candidates[0])
        results_path = next((p for p in results_candidates if os.path.exists(p)), results_candidates[0])
        
        performance = {
            "best_model": "LightGBM",
            "models_trained": 0,
            "metrics": {}
        }
        
        # Load comparison CSV if available
        if os.path.exists(comparison_path):
            try:
                import pandas as pd
                cmp_df = pd.read_csv(comparison_path)
                performance["models_trained"] = len(cmp_df)
                # Get performance metrics
                if 'f1' in cmp_df.columns:
                    best_f1_idx = cmp_df['f1'].idxmax()
                    performance["best_f1"] = float(cmp_df.loc[best_f1_idx, 'f1'])
                    performance["best_model"] = cmp_df.loc[best_f1_idx, 'model']
            except Exception:
                pass
        
        # Load results JSON if available
        if os.path.exists(results_path):
            try:
                import json
                with open(results_path) as f:
                    results = json.load(f)
                performance["results_available"] = True
                # Count models
                performance["models_trained"] = len(results)
            except Exception:
                pass
        
        return performance
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "AI-MFS-Fraud-Detection API",
        "docs": "/docs",
        "version": "1.0.0"
    }


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)