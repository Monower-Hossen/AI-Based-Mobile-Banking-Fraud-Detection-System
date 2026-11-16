#!/usr/bin/env python
# Phase 3: Model Training and Evaluation
# Trains supervised fraud detection models and evaluates them

import pandas as pd
import numpy as np
import time
import json
import os
import sys
from typing import Tuple, Dict, Any, List

import joblib

# Project root detection
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')

sys.path.insert(0, PROJECT_ROOT)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
    auc,
)
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb

from sklearn.ensemble import IsolationForest

from src.data.feature_pipeline import load_data, prepare_features, chronological_split, get_fraud_feature_columns


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed train/val/test data."""
    train_path = os.path.join(PROCESSED_DIR, 'train.parquet')
    val_path = os.path.join(PROCESSED_DIR, 'val.parquet')
    test_path = os.path.join(PROCESSED_DIR, 'test.parquet')
    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path)
    test = pd.read_parquet(test_path)
    return train, val, test


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into feature matrix and target vector.
    
    Drops identifier columns, timestamp, and target-derived features.
    Fills NaN values with column medians.
    """
    # Columns to always exclude as features
    exclude_cols = [
        'transaction_id', 'timestamp', 'sender_id', 'receiver_id',
        'fraud_type', 'is_fraud'
    ]
    
    # Drop identifier columns
    df_features = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors='ignore')
    
    # Select only numeric columns
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    # Remove any that might be target-related
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    X = df_features[numeric_cols].copy()
    
    # Fill NaN values with column medians
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    
    # Ensure no infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    
    y = df['is_fraud'].copy()
    
    return X, y


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series,
                              random_state: int = 42) -> LogisticRegression:
    model = LogisticRegression(
        class_weight='balanced',
        random_state=random_state,
        max_iter=1000,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series,
                        random_state: int = 42) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(
        class_weight='balanced',
        random_state=random_state,
        max_depth=10
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                        random_state: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(
        class_weight='balanced',
        random_state=random_state,
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                  random_state: int = 42) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        class_weight='balanced',
        random_state=random_state,
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series,
                   random_state: int = 42) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective='binary',
        class_weight='balanced',
        random_state=random_state,
        verbose=-1,
        n_estimators=100,
        max_depth=-1,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train, y_train)
    return model


def train_isolation_forest(X_train: pd.DataFrame,
                         random_state: int = 42,
                         contamination: str = 'auto') -> IsolationForest:
    model = IsolationForest(
        random_state=random_state,
        contamination=contamination,
        n_estimators=100,
        max_samples='auto'
    )
    model.fit(X_train)
    return model


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series,
                   name: str = '') -> Dict[str, Any]:
    """Evaluate model and return metrics."""
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X)
    else:
        y_prob = model.predict(X)
    
    y_pred = (y_prob >= 0.5).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average='binary', zero_division=0)
    
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    try:
        roc_auc = roc_auc_score(y, y_prob)
    except Exception:
        roc_auc = 0.0
    
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y, y_prob)
        pr_auc = auc(recall_curve, precision_curve)
    except Exception:
        pr_auc = 0.0
    
    return {
        'name': name,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'false_positive_rate': float(fpr),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }


def tune_threshold(model: Any, X_val: pd.DataFrame, y_val: pd.Series,
                   threshold_range: np.ndarray = np.linspace(0.1, 0.9, 90)) -> Dict[str, Any]:
    """Tune decision threshold using validation data."""
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_val)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X_val)
    else:
        y_prob = model.predict(X_val)
    
    best_f1 = 0
    best_threshold = 0.5
    best_recall = 0
    best_precision = 0
    
    for threshold in threshold_range:
        y_pred = (y_prob >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average='binary', zero_division=0)
        
        if f1 > best_f1 or (f1 == best_f1 and recall > best_recall):
            best_f1 = f1
            best_threshold = threshold
            best_recall = recall
            best_precision = precision
    
    return {
        'best_threshold': float(best_threshold),
        'best_precision': float(best_precision),
        'best_recall': float(best_recall),
        'best_f1': float(best_f1),
    }


def train_and_evaluate_all(
    output_dir: str = 'reports/model_results'
) -> Dict[str, Any]:
    """
    Train all models and evaluate on train/val/test.
    Returns comparison dict and saves models/results.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load processed data
    train, val, test = load_processed_data()
    
    # Split features/target
    X_train, y_train = split_features_target(train)
    X_val, y_val = split_features_target(val)
    X_test, y_test = split_features_target(test)
    
    print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Val: {X_val.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    print(f"Train fraud rate: {y_train.mean():.4f}")
    print(f"Val fraud rate: {y_val.mean():.4f}")
    print(f"Test fraud rate: {y_test.mean():.4f}")
    
    # Scale features - fit only on training data
    scaler = StandardScaler()
    numeric_train_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train[numeric_train_cols] = scaler.fit_transform(X_train[numeric_train_cols])
    X_val[numeric_train_cols] = scaler.transform(X_val[numeric_train_cols])
    X_test[numeric_train_cols] = scaler.transform(X_test[numeric_train_cols])
    
    feature_names = X_train.columns.tolist()
    
    # Models to train
    models = {
        'Logistic Regression': train_logistic_regression,
        'Decision Tree': train_decision_tree,
        'Random Forest': train_random_forest,
        'XGBoost': train_xgboost,
        'LightGBM': train_lightgbm,
    }
    
    results = {}
    best_model_name = None
    best_pr_auc = 0
    
    print("\n=== Training Models ===\n")
    
    for name, train_fn in models.items():
        print(f"Training {name}...")
        t0 = time.time()
        
        model = train_fn(X_train, y_train)
        
        # Evaluate on training data
        train_metrics = evaluate_model(model, X_train, y_train, name=f"{name} (train)")
        
        # Tune threshold on validation data
        threshold_result = tune_threshold(model, X_val, y_val)
        
        # Validation predictions with optimal threshold
        if hasattr(model, 'predict_proba'):
            y_val_prob = model.predict_proba(X_val)[:, 1]
        else:
            y_val_prob = model.decision_function(X_val)
        
        y_val_pred = (y_val_prob >= threshold_result['best_threshold']).astype(int)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            y_val, y_val_pred, average='binary', zero_division=0)
        
        val_roc_auc = roc_auc_score(y_val, y_val_prob) if hasattr(model, 'predict_proba') else 0.0
        
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_val_prob)
            val_pr_auc = auc(recall_curve, precision_curve)
        except Exception:
            val_pr_auc = 0.0
        
        # Test predictions with validation-optimal threshold
        if hasattr(model, 'predict_proba'):
            y_test_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_test_prob = model.decision_function(X_test)
        
        y_test_pred = (y_test_prob >= threshold_result['best_threshold']).astype(int)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            y_test, y_test_pred, average='binary', zero_division=0)
        
        try:
            test_roc_auc = roc_auc_score(y_test, y_test_prob)
        except Exception:
            test_roc_auc = 0.0
        
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_prob)
            test_pr_auc = auc(recall_curve, precision_curve)
        except Exception:
            test_pr_auc = 0.0
        
        # Confusion matrix on test
        cm = confusion_matrix(y_test, y_test_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        result = {
            'train_metrics': train_metrics,
            'val_threshold': threshold_result,
            'val_metrics': {
                'precision': float(val_precision),
                'recall': float(val_recall),
                'f1': float(val_f1),
                'roc_auc': float(val_roc_auc),
                'pr_auc': float(val_pr_auc),
            },
            'test_metrics': {
                'precision': float(test_precision),
                'recall': float(test_recall),
                'f1': float(test_f1),
                'roc_auc': float(test_roc_auc),
                'pr_auc': float(test_pr_auc),
                'false_positive_rate': float(fp / (fp + tn) if (fp + tn) > 0 else 0),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp),
            },
            'best_threshold': threshold_result['best_threshold'],
            'feature_importance': None,
        }
        
        # Get feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            result['feature_importance'] = importance_df.to_dict('records')
            importance_df.head(10).to_csv(
                os.path.join(output_dir, f'{name.lower().replace(" ", "_")}_feature_importance.csv'),
                index=False
            )
        
        results[name] = result
        
        training_time = time.time() - t0
        print(f"  {name}: "
              f"Val F1={val_f1:.4f}, PR-AUC={val_pr_auc:.4f}, "
              f"Test F1={test_f1:.4f}, Test PR-AUC={test_pr_auc:.4f}, "
              f"Time={training_time:.1f}s")
        
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_model_name = name
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f'{name.lower().replace(" ", "_")}_model.pkl')
        joblib.dump(model, model_path)
    
    # Isolation Forest (anomaly detection, unsupervised)
    print("\nTraining Isolation Forest (unsupervised)...")
    iso_model = train_isolation_forest(X_train)
    
    if hasattr(iso_model, 'decision_function'):
        y_val_if_prob = iso_model.decision_function(X_val)
        y_test_if_prob = iso_model.decision_function(X_test)
    else:
        y_val_if_prob = iso_model.score_samples(X_val)
        y_test_if_prob = iso_model.score_samples(X_test)
    
    # For IF, lower score = more anomalous
    val_f1s = []
    thresholds = np.linspace(y_val_if_prob.min(), y_val_if_prob.max(), 50)
    for threshold in thresholds:
        y_val_pred_if = (y_val_if_prob <= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_val, y_val_pred_if, average='binary', zero_division=0)
        val_f1s.append(f1)
    
    best_if_idx = int(np.argmax(val_f1s))
    best_if_threshold = thresholds[best_if_idx]
    
    y_val_pred_if = (y_val_if_prob <= best_if_threshold).astype(int)
    y_test_pred_if = (y_test_if_prob <= best_if_threshold).astype(int)
    
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        y_val, y_val_pred_if, average='binary', zero_division=0)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred_if, average='binary', zero_division=0)
    
    if_cm = confusion_matrix(y_val, y_val_pred_if)
    if_tn, if_fp, if_fn, if_tp = if_cm.ravel() if if_cm.size == 4 else (0, 0, 0, 0)
    
    print(f"  Isolation Forest: Val F1={val_f1:.4f}, Test F1={test_f1:.4f} "
          f"at threshold={best_if_threshold:.4f}")
    
    results['Isolation Forest'] = {
        'train_metrics': {
            'precision': float(val_precision),
            'recall': float(val_recall),
            'f1': float(val_f1),
            'roc_auc': 0.0,
            'pr_auc': 0.0,
        },
        'val_threshold': {'best_threshold': float(best_if_threshold)},
        'val_metrics': {
            'precision': float(val_precision),
            'recall': float(val_recall),
            'f1': float(val_f1),
            'roc_auc': 0.0,
            'pr_auc': 0.0,
        },
        'test_metrics': {
            'precision': float(test_precision),
            'recall': float(test_recall),
            'f1': float(test_f1),
            'roc_auc': 0.0,
            'pr_auc': 0.0,
            'false_positive_rate': float(if_fp / (if_fp + if_tn) if (if_fp + if_tn) > 0 else 0),
            'tn': int(if_tn),
            'fp': int(if_fp),
            'fn': int(if_fn),
            'tp': int(if_tp),
        },
        'best_threshold': float(best_if_threshold),
        'feature_importance': None,
    }
    
    joblib.dump(iso_model, os.path.join(MODEL_DIR, 'isolation_forest_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    
    # Save comparison CSV
    comparison_rows = []
    for name, result in results.items():
        tm = result['test_metrics']
        comparison_rows.append({
            'model': name,
            'precision': tm['precision'],
            'recall': tm['recall'],
            'f1': tm['f1'],
            'roc_auc': tm['roc_auc'],
            'pr_auc': tm['pr_auc'],
            'fpr': tm['false_positive_rate'],
        })
    
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(os.path.join(REPORTS_DIR, 'model_comparison.csv'), index=False)
    print(f"Saved model comparison CSV to {os.path.join(REPORTS_DIR, 'model_comparison.csv')}")
    
    # Save full results as JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (dict, list)):
            return [convert(v) for v in obj] if isinstance(obj, list) else {k: convert(v) for k, v in obj.items()}
        return obj
    
    json_results = {}
    for name, result in results.items():
        json_results[name] = {
            k: convert(v) if not isinstance(v, dict) else {
                k2: convert(v2) for k2, v2 in v.items()
            }
            for k, v in result.items()
        }
    
    with open(os.path.join(REPORTS_DIR, 'results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Saved results JSON to {os.path.join(REPORTS_DIR, 'results.json')}")
    
    print(f"\n=== Best Model: {best_model_name} (PR-AUC: {best_pr_auc:.4f}) ===\n")
    
    return {
        'results': results,
        'best_model': best_model_name,
        'comparison': comparison_df,
    }


if __name__ == '__main__':
    import numpy as np
    
    print("=== Phase 3: Model Training and Evaluation ===\n")
    
    output = train_and_evaluate_all()
    
    print("Phase 3 complete!")
    print(f"Best model: {output['best_model']}")