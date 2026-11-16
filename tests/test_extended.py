"""Tests for feature engineering, model training, and risk scoring."""
import os
import sys
import pandas as pd
import numpy as np

project_root = r"E:\projects\AI-Based-Mobile-Banking-Fraud-Detection-System"


def test_feature_engineering():
    """Test feature engineering pipeline."""
    from src.data.feature_pipeline import (
        create_behavioral_features, create_risk_features,
        create_agent_merchant_risk_features, chronological_split,
        prepare_features
    )
    
    tx = pd.read_parquet(os.path.join(project_root, "data", "raw", "transactions.parquet"))
    
    # Test behavioral features
    tx_beh = create_behavioral_features(tx)
    assert "sender_tx_count_prior" in tx_beh.columns
    assert "sender_amount_mean_prior" in tx_beh.columns
    assert "sender_time_since_prev_tx" in tx_beh.columns
    
    # Test risk features
    tx_risk = create_risk_features(tx_beh)
    assert "device_risk_flag" in tx_risk.columns
    assert "location_risk_flag" in tx_risk.columns
    assert "sim_risk_flag" in tx_risk.columns
    assert "multi_risk_flag" in tx_risk.columns
    
    # Test agent/merchant features
    agents = pd.read_parquet(os.path.join(project_root, "data", "raw", "agents.parquet"))
    merchants = pd.read_parquet(os.path.join(project_root, "data", "raw", "merchants.parquet"))
    tx_features = create_agent_merchant_risk_features(tx_risk, agents, merchants)
    assert "agent_risk_score" in tx_features.columns
    assert "merchant_category" in tx_features.columns
    
    # Test chronological split
    train, val, test = chronological_split(tx)
    n_train = len(train)
    n_val = len(val)
    n_test = len(test)
    assert n_train + n_val + n_test == len(tx)
    assert n_train == 700000, f"Expected 700K train, got {n_train}"
    assert n_val == 150000, f"Expected 150K val, got {n_val}"
    assert n_test == 150000, f"Expected 150K test, got {n_test}"
    
    # Test fraud rates preserved
    assert abs(train["is_fraud"].mean() - 0.05) < 0.001
    assert abs(val["is_fraud"].mean() - 0.05) < 0.001
    assert abs(test["is_fraud"].mean() - 0.05) < 0.001
    
    # Test feature preparation
    X_train, y_train, scaler, le = prepare_features(train, fit_scaler=True)
    assert X_train.shape[1] > 0
    assert y_train.shape[0] == n_train
    assert scaler is not None
    
    print("  PASS: test_feature_engineering")


def test_risk_scoring():
    """Test risk score computation."""
    from src.explainability.shap_explainer import compute_risk_score
    
    # Test various probability values
    for prob in [0.05, 0.25, 0.45, 0.65, 0.85]:
        risk = compute_risk_score(prob)
        assert 0 <= risk["risk_score"] <= 100
        assert risk["risk_category"] in ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
        assert len(risk["review_action"]) > 0
    
    print("  PASS: test_risk_scoring")


def test_model_artifacts():
    """Test that model artifacts exist and are loadable."""
    import joblib
    
    model_path = os.path.join(project_root, "models", "lightgbm_model.pkl")
    scaler_path = os.path.join(project_root, "models", "scaler.pkl")
    
    assert os.path.exists(model_path), f"Model not found at {model_path}"
    os.path.exists(scaler_path), f"Scaler not found at {scaler_path}"
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    assert hasattr(model, "predict")
    assert hasattr(scaler, "transform")
    
    print("  PASS: test_model_artifacts")


if __name__ == "__main__":
    test_feature_engineering()
    test_risk_scoring()
    test_model_artifacts()
    print("\nAll extended tests passed!")