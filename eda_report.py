import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'raw')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'reports', 'eda')

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    tx = pd.read_parquet(os.path.join(DATA_DIR, 'transactions.parquet'))
    cust = pd.read_parquet(os.path.join(DATA_DIR, 'customers.parquet'))
    agents = pd.read_parquet(os.path.join(DATA_DIR, 'agents.parquet'))
    merchants = pd.read_parquet(os.path.join(DATA_DIR, 'merchants.parquet'))
    return tx, cust, agents, merchants


def inspect_duplicates_missing(tx, cust, agents, merchants):
    results = {}
    results['duplicates'] = {
        'transactions': int(tx.duplicated().sum()),
        'customers': int(cust.duplicated().sum()),
        'agents': int(agents.duplicated().sum()),
        'merchants': int(merchants.duplicated().sum()),
    }
    results['missing'] = {
        'transactions': {
            'total': int(tx.isnull().sum().sum()),
            'by_column': {c: int(tx[c].isnull().sum()) for c in tx.columns if tx[c].isnull().any()}
        },
        'customers': {
            'total': int(cust.isnull().sum().sum()),
            'by_column': {c: int(cust[c].isnull().sum()) for c in cust.columns if cust[c].isnull().any()}
        },
        'agents': {
            'total': int(agents.isnull().sum().sum()),
            'by_column': {c: int(agents[c].isnull().sum()) for c in agents.columns if agents[c].isnull().any()}
        },
        'merchants': {
            'total': int(merchants.isnull().sum().sum()),
            'by_column': {c: int(merchants[c].isnull().sum()) for c in merchants.columns if merchants[c].isnull().any()}
        },
    }
    return results


def inspect_fraud_distribution(tx):
    results = {}
    results['fraud_rate'] = float(tx['is_fraud'].mean())
    results['fraud_counts'] = {
        'non_fraud': int((tx['is_fraud'] == 0).sum()),
        'fraud': int((tx['is_fraud'] == 1).sum()),
    }
    results['fraud_type_distribution'] = {
        k: int(v) for k, v in tx['fraud_type'].value_counts().to_dict().items()
    }
    results['fraud_scenarios'] = [
        'account_takeover', 'sim_swap', 'rapid_transaction',
        'money_laundering', 'account_mule', 'transaction_splitting',
        'suspicious_cash_out', 'unusual_location', 'new_device', 'agent_collusion'
    ]
    return results


def inspect_temporal_distribution(tx):
    tx_sorted = tx.sort_values('timestamp').copy()
    results = {}
    results['timestamp_min'] = tx_sorted['timestamp'].min().isoformat()
    results['timestamp_max'] = tx_sorted['timestamp'].max().isoformat()
    results['timestamp_range_days'] = (tx_sorted['timestamp'].max() - tx_sorted['timestamp'].min()).days
    results['transactions_per_day'] = len(tx_sorted) / max(results['timestamp_range_days'], 1)
    return results


def inspect_feature_distributions(tx):
    results = {}
    
    # Amount distribution
    fig, ax = plt.subplots()
    tx['amount_bdt'].hist(bins=50, ax=ax, alpha=0.7)
    ax.set_title('Transaction Amount Distribution')
    ax.set_xlabel('Amount (BDT)')
    ax.set_ylabel('Frequency')
    fig.savefig(os.path.join(OUTPUT_DIR, 'amount_distribution.png'))
    plt.close(fig)
    results['amount_stats'] = {
        'min': float(tx['amount_bdt'].min()),
        'max': float(tx['amount_bdt'].max()),
        'mean': float(tx['amount_bdt'].mean()),
        'median': float(tx['amount_bdt'].median()),
    }
    
    # Hour distribution
    fig, ax = plt.subplots()
    tx['hour'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_title('Transaction Hour Distribution')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Count')
    fig.savefig(os.path.join(OUTPUT_DIR, 'hour_distribution.png'))
    plt.close(fig)
    
    # Day of week distribution
    fig, ax = plt.subplots()
    tx['day_of_week'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_title('Transaction Day of Week Distribution')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Count')
    fig.savefig(os.path.join(OUTPUT_DIR, 'day_of_week_distribution.png'))
    plt.close(fig)
    
    # Is night distribution
    fig, ax = plt.subplots()
    tx['is_night'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_title('Night Transaction Distribution')
    ax.set_xlabel('Is Night')
    ax.set_ylabel('Count')
    fig.savefig(os.path.join(OUTPUT_DIR, 'is_night_distribution.png'))
    plt.close(fig)
    
    # Fraud amount distribution
    fig, ax = plt.subplots()
    tx[tx['is_fraud'] == 1]['amount_bdt'].hist(bins=50, ax=ax, alpha=0.7, color='red')
    tx[tx['is_fraud'] == 0]['amount_bdt'].hist(bins=50, ax=ax, alpha=0.3, color='blue')
    ax.set_title('Amount Distribution by Fraud Status')
    ax.set_xlabel('Amount (BDT)')
    ax.set_ylabel('Frequency')
    ax.legend(['Fraud', 'Non-Fraud'])
    fig.savefig(os.path.join(OUTPUT_DIR, 'amount_by_fraud.png'))
    plt.close(fig)
    
    # Transaction type distribution
    fig, ax = plt.subplots()
    tx['transaction_type'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Transaction Type Distribution')
    ax.set_xlabel('Transaction Type')
    ax.set_ylabel('Count')
    fig.savefig(os.path.join(OUTPUT_DIR, 'transaction_type_distribution.png'))
    plt.close(fig)
    
    # Network type distribution
    fig, ax = plt.subplots()
    tx['network_type'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Network Type Distribution')
    ax.set_xlabel('Network Type')
    ax.set_ylabel('Count')
    fig.savefig(os.path.join(OUTPUT_DIR, 'network_type_distribution.png'))
    plt.close(fig)
    
    # Device type distribution
    fig, ax = plt.subplots()
    tx['device_type'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Device Type Distribution')
    ax.set_xlabel('Device Type')
    ax.set_ylabel('Count')
    fig.savefig(os.path.join(OUTPUT_DIR, 'device_type_distribution.png'))
    plt.close(fig)
    
    # SIM changed distribution
    fig, ax = plt.subplots()
    tx['sim_changed'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('SIM Changed Distribution')
    ax.set_xlabel('SIM Changed')
    ax.set_ylabel('Count')
    fig.savefig(os.path.join(OUTPUT_DIR, 'sim_changed_distribution.png'))
    plt.close(fig)
    
    # New device distribution
    fig, ax = plt.subplots()
    tx['new_device'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('New Device Distribution')
    ax.set_xlabel('New Device')
    ax.set_ylabel('Count')
    fig.savefig(os.path.join(OUTPUT_DIR, 'new_device_distribution.png'))
    plt.close(fig)
    
    # Unusual location distribution
    fig, ax = plt.subplots()
    tx['unusual_location'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Unusual Location Distribution')
    ax.set_xlabel('Unusual Location')
    ax.set_ylabel('Count')
    fig.savefig(os.path.join(OUTPUT_DIR, 'unusual_location_distribution.png'))
    plt.close(fig)
    
    # Failed login attempts distribution
    fig, ax = plt.subplots()
    tx['failed_login_attempts'].hist(bins=10, ax=ax, alpha=0.7)
    ax.set_title('Failed Login Attempts Distribution')
    ax.set_xlabel('Failed Login Attempts')
    ax.set_ylabel('Count')
    fig.savefig(os.path.join(OUTPUT_DIR, 'failed_login_attempts_distribution.png'))
    plt.close(fig)
    
    return results


def inspect_correlations(tx):
    numeric_cols = tx.select_dtypes(include=[np.number]).columns
    corr = tx[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    fig.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'))
    plt.close(fig)
    
    # Specifically correlate fraud-related features
    results = {}
    fraud_corr = corr['is_fraud'].sort_values(ascending=False)
    results['feature_correlations_with_fraud'] = {
        col: float(val) for col, val in fraud_corr.items() if col != 'is_fraud'
    }
    return results


def generate_report(tx, cust, agents, merchants):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    report = {
        'generated_at': timestamp,
        'dataset': {
            'transactions': {
                'rows': int(tx.shape[0]),
                'columns': int(tx.shape[1]),
            },
            'customers': {
                'rows': int(cust.shape[0]),
                'columns': int(cust.shape[1]),
            },
            'agents': {
                'rows': int(agents.shape[0]),
                'columns': int(agents.shape[1]),
            },
            'merchants': {
                'rows': int(merchants.shape[0]),
                'columns': int(merchants.shape[1]),
            },
        },
    }
    
    report.update(inspect_duplicates_missing(tx, cust, agents, merchants))
    report.update(inspect_fraud_distribution(tx))
    report.update(inspect_temporal_distribution(tx))
    report.update(inspect_feature_distributions(tx))
    report.update(inspect_correlations(tx))
    
    report_path = os.path.join(OUTPUT_DIR, f'data_quality_report_{timestamp}.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to: {report_path}")
    return report


if __name__ == '__main__':
    print("Loading data...")
    tx, cust, agents, merchants = load_data()
    print(f"Transactions: {tx.shape[0]} rows")
    print(f"Customers: {cust.shape[0]} rows")
    print(f"Agents: {agents.shape[0]} rows")
    print(f"Merchants: {merchants.shape[0]} rows")
    
    print("\nGenerating EDA report and visualizations...")
    report = generate_report(tx, cust, agents, merchants)
    
    print("\nPhase 1 complete. Files created:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  - {f}")