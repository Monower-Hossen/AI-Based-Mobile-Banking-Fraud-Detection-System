#!/usr/bin/env python
# Phase 4: Explainable AI & Risk Scoring
# SHAP-based explanations and fraud risk scoring

import numpy as np
import pandas as pd
import os
import sys
from typing import Tuple, Dict, Any, List, Optional

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import shap

from src.models.train import split_features_target


def compute_shap_values(model: Any, X: pd.DataFrame,
                        feature_names: List[str],
                        model_type: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values for the given model and data.
    
    Uses appropriate SHAP explainer based on model type.
    Returns shape (n_samples, n_features) so that
    shap_values[instance_idx] gives SHAP values for all features.
    """
    try:
        # Determine the best explainer for the model type
        if model_type == 'auto':
            # Try TreeExplainer first for tree-based models
            if hasattr(model, 'feature_importances_'):
                model_type = 'tree'
            else:
                model_type = 'linear'
        
        if model_type == 'tree':
            # For tree-based models (LightGBM, XGBoost, Random Forest, Decision Tree)
            # Use TreeExplainer with tree_path_dependent perturbation
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            shap_values = explainer.shap_values(X)
            # shap_values shape for binary tree models: (n_samples, n_features)
            # or list of per-class arrays
            if isinstance(shap_values, list):
                # Multiple classes - take index 1 if available, else index 0
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Ensure 2D: (n_samples, n_features)
            if shap_values.ndim == 1:
                # If 1D reshape to (n_samples, 1)
                shap_values = shap_values.reshape(-1, 1)
            
            # Don't squeeze features dimension - keep all features
            # The shape should be (n_samples, n_features)
            
        elif model_type == 'linear':
            # For logistic regression and similar
            explainer = shap.LinearExplainer(X, model, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif shap_values.ndim == 2:
                # Take positive class column if available
                if shap_values.shape[1] > 1:
                    shap_values = shap_values[:, 1]
                else:
                    shap_values = shap_values[:, 0]
            
            # Ensure 2D
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(-1, 1)
        
        else:
            # Fallback: try TreeExplainer regardless
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Ensure 2D
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(-1, 1)
        
        # At this point shap_values should be (n_samples, n_features_or_1)
        # We DON'T squeeze the feature dimension - keep it so that
        # shap_values[instance_idx] returns values for all features
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(-1, 1)
        
        # Return the full shap_values matrix (n_samples, n_features)
        # If somehow we have more columns than feature_names, truncate/pad
        n_features = max(1, len(feature_names))
        if shap_values.shape[1] > n_features:
            # If more SHAP columns than feature names, take first n_features
            shap_values = shap_values[:, :n_features]
        elif shap_values.shape[1] < n_features:
            # If fewer SHAP columns, pad with zeros
            padding = np.zeros((shap_values.shape[0], n_features - shap_values.shape[1]))
            shap_values = np.hstack([shap_values, padding])
        
        return shap_values, np.arange(n_features)
    
    except Exception as e:
        print(f"  SHAP value computation failed: {e}")
        # Return zeros with shape (n_samples, n_features)
        n_features = max(1, len(feature_names))
        return np.zeros((X.shape[0], n_features)), np.arange(n_features)


def generate_global_importance(model: Any, X: pd.DataFrame,
                               feature_names: List[str]) -> pd.DataFrame:
    """Generate global SHAP feature importance."""
    try:
        shap_values, _ = compute_shap_values(model, X, feature_names)
        # Mean absolute SHAP value per feature across all samples
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_shap_abs': mean_abs_shap
        }).sort_values('mean_shap_abs', ascending=False)
        return importance_df
    except Exception as e:
        print(f"  Global importance generation failed: {e}")
        return pd.DataFrame({'feature': feature_names, 'mean_shap_abs': np.zeros(len(feature_names))})


def generate_local_explanation(model: Any, X: pd.DataFrame,
                               feature_names: List[str],
                               instance_idx: int) -> Dict[str, Any]:
    """Generate local SHAP explanation for a single instance."""
    try:
        shap_values, _ = compute_shap_values(model, X, feature_names)
        
        # Get SHAP values for this specific instance
        # shap_values shape: (n_samples, n_features)
        # shap_values[instance_idx] gives SHAP values for all features of this instance
        if instance_idx < 0 or instance_idx >= shap_values.shape[0]:
            return {
                'instance_idx': instance_idx,
                'top_positive_features': [],
                'top_negative_features': [],
                'shap_values_dict': {}
            }
        
        instance_shap = shap_values[instance_idx]
        
        # Get indices sorted by SHAP value (highest to lowest)
        sorted_indices = np.argsort(instance_shap)[::-1]
        
        # Separate positive and negative contributing features
        pos_features = []
        neg_features = []
        
        for i in sorted_indices:
            val = float(instance_shap[i])
            name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            if val > 0 and len(pos_features) < 5:
                pos_features.append((name, val))
            elif val < 0 and len(neg_features) < 5:
                neg_features.append((name, val))
        
        # Build SHAP values dict for features with non-zero impact
        shap_values_dict = {}
        for i in range(len(feature_names)):
            abs_val = abs(instance_shap[i]) if i < len(instance_shap) else 0
            if abs_val > 0.01:  # threshold for significance
                shap_values_dict[feature_names[i]] = float(instance_shap[i])
        
        return {
            'instance_idx': instance_idx,
            'top_positive_features': pos_features,
            'top_negative_features': neg_features,
            'shap_values_dict': shap_values_dict
        }
    except Exception as e:
        print(f"  Local explanation generation failed: {e}")
        return {
            'instance_idx': instance_idx,
            'top_positive_features': [],
            'top_negative_features': [],
            'shap_values_dict': {}
        }


def compute_risk_score(fraud_probability: float,
                       low_threshold: float = 0.3,
                       high_threshold: float = 0.7,
                       critical_threshold: float = 0.9,
                       score_max: int = 100) -> Dict[str, Any]:
    """Convert fraud probability to interpretable risk score 0-100.
    
    Risk levels:
    - 0-29: Low Risk
    - 30-59: Medium Risk
    - 60-79: High Risk
    - 80-100: Critical Risk
    """
    try:
        prob = float(fraud_probability)
        
        if prob >= critical_threshold:
            risk_score = int(score_max * (prob - critical_threshold) / (1.0 - critical_threshold))
            risk_score = min(risk_score, score_max)
            risk_category = "Critical Risk"
            review_action = "Immediate review required - flag for manual inspection"
        elif prob >= high_threshold:
            risk_score = int(score_max * (prob - high_threshold) / (critical_threshold - high_threshold)) + 60
            risk_score = min(max(risk_score, 60), 79)
            risk_category = "High Risk"
            review_action = "Priority review - monitor closely"
        elif prob >= low_threshold:
            risk_score = int(score_max * (prob - low_threshold) / (high_threshold - low_threshold)) + 30
            risk_score = min(max(risk_score, 30), 59)
            risk_category = "Medium Risk"
            review_action = "Standard review - verify transaction"
        else:
            risk_score = int(score_max * prob / low_threshold)
            risk_score = min(max(risk_score, 0), 29)
            risk_category = "Low Risk"
            review_action = "No review needed - transaction is routine"
        
        return {
            'fraud_probability': prob,
            'risk_score': risk_score,
            'risk_category': risk_category,
            'review_action': review_action,
            'low_threshold': low_threshold,
            'high_threshold': high_threshold,
            'critical_threshold': critical_threshold,
        }
    except Exception as e:
        return {
            'fraud_probability': 0.0,
            'risk_score': 0,
            'risk_category': 'Low Risk',
            'review_action': 'Unable to compute risk',
        }


def format_shap_local_explanation(shap_explanation: Dict[str, Any]) -> str:
    """Format SHAP local explanation as readable text."""
    lines = []
    lines.append(f"Instance #{shap_explanation['instance_idx']}")
    prob = shap_explanation.get('predicted_fraud_probability', 0)
    lines.append(f"Fraud Probability: {prob:.4f}")
    lines.append("")
    
    if shap_explanation.get('top_positive_features'):
        lines.append("Features increasing fraud risk:")
        for feat, val in shap_explanation['top_positive_features']:
            lines.append(f"  - {feat}: +{val:.4f}")
    
    if shap_explanation.get('top_negative_features'):
        lines.append("Features decreasing fraud risk:")
        for feat, val in shap_explanation['top_negative_features']:
            lines.append(f"  - {feat}: {val:.4f} (protective)")
    
    return "\n".join(lines)


def run_shap_analysis(model: Any, X: pd.DataFrame,
                      feature_names: List[str],
                      max_instances: int = 5,
                      subsample_size: int = 100) -> Dict[str, Any]:
    """Run complete SHAP analysis on a model.
    
    For large datasets, subsamples for SHAP computation for performance.
    """
    print(f"  Running SHAP analysis (max {max_instances} instances, subsample {subsample_size} if needed)...")
    
    # For large datasets, subsample for SHAP computation
    if len(X) > subsample_size:
        np.random.seed(42)
        sample_indices = np.random.choice(len(X), subsample_size, replace=False)
        X_sample = X.iloc[sample_indices]
        print(f"  Using subsample of {subsample_size} instances from {len(X)} total")
    else:
        X_sample = X
        print(f"  Using all {len(X)} instances")
    
    # Compute SHAP values
    shap_values, _ = compute_shap_values(model, X_sample, feature_names)
    
    results = {
        'explainer_type': 'shap',
        'global_importance': None,
        'local_explanations': [],
    }
    
    # Global importance
    global_imp = generate_global_importance(model, X_sample, feature_names)
    results['global_importance'] = global_imp.to_dict('records')
    
    # Local explanations for sampled instances
    if len(X) > subsample_size:
        sample_indices = np.random.choice(len(X), max_instances, replace=False)
    else:
        sample_indices = np.random.choice(len(X), max_instances, replace=False)
    
    for idx in sample_indices:
        # Find the relative index within X_sample
        if len(X) > subsample_size:
            rel_idx = np.where(sample_indices == idx)[0][0]
        else:
            rel_idx = idx
        
        # Ensure rel_idx is valid
        rel_idx = min(rel_idx, shap_values.shape[0] - 1) if 'shap_values' in dir() else 0
        
        local = generate_local_explanation(model, X_sample, feature_names, rel_idx)
        local['original_index'] = idx  # track original data index
        results['local_explanations'].append(local)
    
    return results


if __name__ == '__main__':
    import json
    
    print("=== Phase 4: Explainable AI & Risk Scoring ===\n")
    
    # Load best model (LightGBM)
    model_path = 'models/lightgbm_model.pkl'
    if os.path.exists(model_path):
        import joblib
        model = joblib.load(model_path)
        print(f"Loaded model: {model_path}")
    else:
        print(f"Model not found at {model_path}")
        for f in sorted(os.listdir('models')):
            if f.endswith('_model.pkl') and f != 'scaler.pkl' and f != 'isolation_forest_model.pkl':
                import joblib
                model = joblib.load(os.path.join('models', f))
                model_path = os.path.join('models', f)
                break
    
    # Load processed data
    train = pd.read_parquet('data/processed/train.parquet')
    X_train, y_train = split_features_target(train)
    feature_names = X_train.columns.tolist()
    
    # Run SHAP analysis
    shap_results = run_shap_analysis(model, X_train, feature_names, max_instances=3, subsample_size=100)
    
    print("\n  Global Feature Importance (SHAP):")
    if shap_results['global_importance']:
        for row in shap_results['global_importance'][:10]:
            print(f"    {row['feature']}: {row['mean_shap_abs']:.4f}")
    
    print("\n  Local Explanations (sample instances):")
    for local in shap_results['local_explanations']:
        print(format_shap_local_explanation(local))
        print()
    
    # Risk scoring example
    print("  Risk Score Examples:")
    for prob in [0.05, 0.25, 0.45, 0.65, 0.85]:
        risk = compute_risk_score(prob)
        print(f"    Probability {prob:.2f} -> Risk Score {risk['risk_score']} ({risk['risk_category']}): {risk['review_action']}")
    
    print("\nPhase 4 complete!")