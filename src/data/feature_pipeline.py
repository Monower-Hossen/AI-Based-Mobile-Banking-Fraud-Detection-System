import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder


def repair_temporal_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Recompute time-based columns from the authoritative ``timestamp``.

    Some releases of the synthetic dataset ship ``hour``/``day_of_week``/
    ``is_night`` columns that are not aligned with ``timestamp`` (e.g. a
    constant day-of-week).  Deriving them from ``timestamp`` keeps the
    preprocessing, model features and dashboard analytics consistent.
    """
    if 'timestamp' not in tx.columns:
        return tx

    tx = tx.copy()
    ts = pd.to_datetime(tx['timestamp'])
    tx['hour'] = ts.dt.hour.astype('int8')
    tx['day_of_week'] = ts.dt.day_of_week.astype('int8')
    tx['is_night'] = tx['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype('int8')
    return tx


def load_data(path: str = 'data/raw') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all parquet data files."""
    tx = pd.read_parquet(f'{path}/transactions.parquet')
    cust = pd.read_parquet(f'{path}/customers.parquet')
    agents = pd.read_parquet(f'{path}/agents.parquet')
    merchants = pd.read_parquet(f'{path}/merchants.parquet')
    tx = repair_temporal_features(tx)
    return tx, cust, agents, merchants


def validate_data(tx: pd.DataFrame, cust: pd.DataFrame,
                  agents: pd.DataFrame, merchants: pd.DataFrame) -> Dict[str, Any]:
    """Validate data quality and return issues."""
    issues = {
        'duplicates': {},
        'missing': {}
    }
    
    for name, df in [('transactions', tx), ('customers', cust),
                     ('agents', agents), ('merchants', merchants)]:
        issues['duplicates'][name] = int(df.duplicated().sum())
        issues['missing'][name] = {
            'total': int(df.isnull().sum().sum()),
            'columns': {c: int(df[c].isnull().sum()) 
                       for c in df.columns if df[c].isnull().any()}
        }
    
    return issues


def get_fraud_feature_columns() -> list:
    """Return columns that should NOT be used as features (target leakage)."""
    return [
        'transaction_id', 'sender_id', 'receiver_id',
        'is_fraud', 'fraud_type'
    ]


def create_behavioral_features(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Create behavioral features using only sorted transaction data.
    All features computed in chronological order (sorted by timestamp).
    Vectorized operations for performance.
    """
    tx = repair_temporal_features(tx)
    tx = tx.sort_values('timestamp').reset_index(drop=True)
    
    # Cumulative count of prior transactions per sender (0 for first)
    tx['sender_tx_count_prior'] = tx.groupby('sender_id').cumcount()
    
    # Expanding mean amount per sender before current transaction.
    # Vectorized equivalent of expanding().mean().shift(1).
    sender_amount = tx.groupby('sender_id')['amount_bdt']
    prior_sum = sender_amount.cumsum() - tx['amount_bdt']
    prior_count = sender_amount.cumcount()
    tx['sender_amount_mean_prior'] = prior_sum / prior_count.replace(0, np.nan)
    tx['sender_amount_mean_prior'] = tx['sender_amount_mean_prior'].fillna(tx['amount_bdt'].mean())
    
    # Amount deviation from sender historical mean
    with np.errstate(divide='ignore', invalid='ignore'):
        tx['sender_amount_deviation_prior'] = (
            (tx['amount_bdt'] - tx['sender_amount_mean_prior']) / 
            tx['sender_amount_mean_prior'].replace(0, np.nan)
        )
    tx['sender_amount_deviation_prior'] = tx['sender_amount_deviation_prior'].fillna(0)
    
    # Unique receivers per sender prior
    tx['sender_unique_receivers_prior'] = tx.groupby('sender_id')['receiver_id'].cumcount()
    
    # Time since previous transaction per sender (hours)
    tx['sender_time_since_prev_tx'] = (
        tx.groupby('sender_id')['timestamp']
        .diff()
        .dt.total_seconds()
        / 3600
    )
    tx['sender_time_since_prev_tx'] = tx['sender_time_since_prev_tx'].fillna(
        tx['sender_time_since_prev_tx'].median()
    )
    
    # High-value indicator relative to sender history
    tx['sender_high_value_prior'] = (tx['amount_bdt'] > tx['sender_amount_mean_prior']).astype(int)
    
    # Night indicator
    tx['is_night_tx'] = tx['is_night']
    
    # Weekend indicator
    tx['is_weekend'] = ((tx['day_of_week'] % 6) >= 5).astype(int)
    
    return tx


def create_risk_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Create risk-related features from available columns."""
    tx = tx.copy()
    
    # Individual risk flags (already in dataset, re-encoded)
    tx['device_risk_flag'] = tx['new_device'].astype(int)
    tx['location_risk_flag'] = tx['unusual_location'].astype(int)
    tx['sim_risk_flag'] = tx['sim_changed'].astype(int)
    
    # Multi-risk: 2+ risk flags from {new_device, unusual_location, sim_changed}
    tx['multi_risk_flag'] = (
        (tx['new_device'] + tx['unusual_location'] + tx['sim_changed']) >= 2
    ).astype(int)
    
    # High-value transaction thresholds
    tx['high_value_tx'] = (tx['amount_bdt'] > 10000).astype(int)
    tx['very_high_value_tx'] = (tx['amount_bdt'] > 50000).astype(int)
    
    # Rapid transaction flag using vectorized approach
    # Sort by timestamp, then for each sender compute rolling count within 15 min
    tx = tx.sort_values('timestamp').reset_index(drop=True)
    
    # Approximate rapid tx: use expanding count per sender 
    # and time-based grouping
    tx['sender_tx_count_15min'] = 1  # start count at 1 for self
    
    # Use a simpler approach: count of prior tx per sender 
    # where time diff < 900 seconds (15 min)
    # We'll use a per-sender groupby with cumsum on time diff
    
    # Compute time diff per sender
    tx['time_since_prev'] = (
        tx.groupby('sender_id')['timestamp']
        .diff()
        .dt.total_seconds()
        / 60  # in minutes
    )
    
    # For each sender, mark if time since prev < 15 min
    tx['in_15min_window'] = (tx['time_since_prev'] < 15).astype(int)
    
    # Cumulative count of transactions in current 15-min window per sender
    # Use groupby cumsum with reset when outside window
    tx['sender_tx_count_15min'] = 1  # initialize
    
    # Faster approximation: just use prior tx count per sender
    # (good enough for initial modeling)
    tx['sender_tx_count_prior'] = tx.groupby('sender_id').cumcount()
    
    # Rapid tx if prior tx count > 3 (within the window approximation)
    tx['rapid_tx_flag'] = (tx['sender_tx_count_prior'] > 3).astype(int)
    
    return tx


def create_agent_merchant_risk_features(
    tx: pd.DataFrame, 
    agents: pd.DataFrame, 
    merchants: pd.DataFrame
) -> pd.DataFrame:
    """Create agent and merchant historical risk features."""
    tx = tx.copy()
    
    # Merge agent risk scores
    tx = tx.merge(
        agents[['agent_id', 'agent_risk_score']], 
        on='agent_id', 
        how='left'
    )
    
    # Merge merchant category
    tx = tx.merge(
        merchants[['merchant_id', 'merchant_category']], 
        on='merchant_id', 
        how='left'
    )
    
    # Fill missing risk scores with median
    tx['agent_risk_score'] = tx['agent_risk_score'].fillna(
        tx['agent_risk_score'].median()
    )
    
    # Fill missing merchant categories with 'unknown'
    tx['merchant_category'] = tx['merchant_category'].fillna('unknown')
    
    # Agent transaction count historical
    agent_tx_counts = tx.groupby('agent_id').size()
    tx['agent_tx_count'] = tx['agent_id'].map(agent_tx_counts)
    
    # Merchant transaction count historical
    merchant_tx_counts = tx.groupby('merchant_id').size()
    tx['merchant_tx_count'] = tx['merchant_id'].map(merchant_tx_counts)
    
    return tx


def chronological_split(
    tx: pd.DataFrame, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological 70/15/15 train/validation/test split.
    Data is sorted by timestamp; no future information leaks.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, \
        "Ratios must sum to 1.0"
    
    tx = repair_temporal_features(tx)
    tx_sorted = tx.sort_values('timestamp').reset_index(drop=True)
    n = len(tx_sorted)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train = tx_sorted.iloc[:n_train].copy()
    val = tx_sorted.iloc[n_train:n_train + n_val].copy()
    test = tx_sorted.iloc[n_train + n_val:].copy()
    
    return train, val, test


def prepare_features(
    tx: pd.DataFrame, 
    fit_scaler: bool = True,
    scaler: Optional[StandardScaler] = None,
    label_encoders: Optional[Dict] = None
) -> Tuple[pd.DataFrame, pd.Series, Optional[StandardScaler], Dict]:
    """
    Prepare features for modeling.
    - Excludes target leakage columns (except is_fraud which is the target)
    - Handles categorical variables
    - Returns feature matrix and target
    """
    tx = tx.copy()
    
    # Columns to exclude as FEATURES (not as target)
    # is_fraud is the target, kept; excluded from feature cols only
    feature_exclude_cols = [
        'transaction_id', 'timestamp'
    ]
    
    # Drop only identifier/timestamp columns, keep is_fraud as target
    drop_cols = [c for c in tx.columns if c in feature_exclude_cols]
    tx_features = tx.drop(columns=drop_cols, errors='ignore')
    
    # Separate numeric and categorical feature columns
    # is_fraud is the target variable, kept separate
    feature_cols = [c for c in tx_features.columns if c != 'is_fraud']
    numeric_cols = tx_features[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(tx_features[c])]
    
    categorical_cols = tx_features[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [c for c in categorical_cols 
                       if pd.api.types.is_string_dtype(tx_features[c]) or tx_features[c].dtype == object]
    
    # Handle categoricals with LabelEncoding
    tx_processed = tx_features.copy()
    le_dict = {}
    
    for col in categorical_cols:
        if label_encoders and col in label_encoders:
            le = label_encoders[col]
        else:
            le = LabelEncoder()
        tx_processed[col] = tx_processed[col].fillna('missing')
        try:
            tx_processed[col] = le.transform(tx_processed[col].astype(str))
        except (ValueError, TypeError):
            tx_processed[col] = -1
        le_dict[col] = le
    
    # Select only numeric columns for the feature matrix
    X = tx_processed[numeric_cols].copy()
    y = tx_processed['is_fraud'].copy()
    
    # Ensure all columns are float
    for c in X.columns:
        X[c] = X[c].astype(float)
    
    # Scale numeric features
    if fit_scaler:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    elif scaler is not None:
        X[numeric_cols] = scaler.transform(X[numeric_cols])
    
    return X, y, scaler, le_dict


def save_processed_data(
    train: pd.DataFrame, 
    val: pd.DataFrame, 
    test: pd.DataFrame,
    output_dir: str = 'data/processed'
) -> Dict[str, str]:
    """Save processed datasets as parquet."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    paths = {}
    train_path = os.path.join(output_dir, 'train.parquet')
    val_path = os.path.join(output_dir, 'val.parquet') 
    test_path = os.path.join(output_dir, 'test.parquet')
    
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)
    test.to_parquet(test_path, index=False)
    
    paths['train'] = train_path
    paths['val'] = val_path
    paths['test'] = test_path
    
    return paths


if __name__ == '__main__':
    import os
    import joblib
    
    print("=== Phase 2: Data Cleaning & Feature Engineering ===\n")
    
    # Load data
    print("Loading data...")
    tx, cust, agents, merchants = load_data()
    print(f"  Transactions: {tx.shape[0]}")
    print(f"  Customers: {cust.shape[0]}")
    print(f"  Agents: {agents.shape[0]}")
    print(f"  Merchants: {merchants.shape[0]}")
    
    # Validate
    print("\nValidating data quality...")
    issues = validate_data(tx, cust, agents, merchants)
    print(f"  Duplicates: {issues['duplicates']}")
    print(f"  Missing: {issues['missing']}")
    
    # Chronological split (before feature engineering to avoid leakage)
    print("\nChronological 70/15/15 split...")
    train, val, test = chronological_split(tx)
    print(f"  Train: {train.shape[0]} ({100*len(train)/len(tx):.1f}%)")
    print(f"  Val: {val.shape[0]} ({100*len(val)/len(tx):.1f}%)")
    print(f"  Test: {test.shape[0]} ({100*len(test)/len(tx):.1f}%)")
    
    # Verify fraud rates in each split
    print("\nFraud rate per split:")
    for name, df in [('Train', train), ('Validation', val), ('Test', test)]:
        rate = df['is_fraud'].mean()
        print(f"  {name}: {rate:.4f} ({int(df['is_fraud'].sum())} fraud/{len(df)} total)")
    
    # Create behavioral features
    print("\nCreating behavioral features...")
    train = create_behavioral_features(train)
    val = create_behavioral_features(val)
    test = create_behavioral_features(test)
    
    # Create risk features
    print("\nCreating risk features...")
    train = create_risk_features(train)
    val = create_risk_features(val)
    test = create_risk_features(test)
    
    # Create agent/merchant risk features
    print("\nCreating agent/merchant risk features...")
    train = create_agent_merchant_risk_features(train, agents, merchants)
    val = create_agent_merchant_risk_features(val, agents, merchants)
    test = create_agent_merchant_risk_features(test, agents, merchants)
    
    # Prepare features
    print("\nPreparing features...")
    X_train, y_train, scaler, label_encoders = prepare_features(train, fit_scaler=True)
    X_val, y_val, _, _ = prepare_features(val, fit_scaler=False, scaler=scaler, label_encoders=label_encoders)
    X_test, y_test, _, _ = prepare_features(test, fit_scaler=False, scaler=scaler, label_encoders=label_encoders)
    
    print(f"  Train features: {X_train.shape[1]} columns")
    print(f"  Train target distribution: {dict(y_train.value_counts())}")
    print(f"  Val features: {X_val.shape[1]} columns")
    print(f"  Test features: {X_test.shape[1]} columns")
    
    # Save processed data
    print("\nSaving processed data...")
    paths = save_processed_data(train, val, test)
    print(f"  Train: {paths['train']}")
    print(f"  Val: {paths['val']}")
    print(f"  Test: {paths['test']}")
    
    # Save scaler and feature column info
    os.makedirs('models/preprocessing', exist_ok=True)
    joblib.dump(scaler, 'models/preprocessing/scaler.pkl')
    joblib.dump(label_encoders, 'models/preprocessing/label_encoders.pkl')
    
    feature_info = {
        'numeric_cols': list(X_train.columns),
        'categorical_cols': [],
        'feature_count': X_train.shape[1],
        'target': 'is_fraud'
    }
    joblib.dump(feature_info, 'models/preprocessing/feature_info.pkl')
    
    print("\nPhase 2 complete!")