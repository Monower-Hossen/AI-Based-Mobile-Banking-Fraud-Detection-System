"""Tests for data loading and validation."""
import os
import sys
import pandas as pd


def test_data_loading():
    """Test that data files can be loaded."""
    project_root = r"E:\projects\AI-Based-Mobile-Banking-Fraud-Detection-System"
    tx = pd.read_parquet(os.path.join(project_root, "data", "raw", "transactions.parquet"))
    cust = pd.read_parquet(os.path.join(project_root, "data", "raw", "customers.parquet"))
    agents = pd.read_parquet(os.path.join(project_root, "data", "raw", "agents.parquet"))
    merchants = pd.read_parquet(os.path.join(project_root, "data", "raw", "merchants.parquet"))
    assert tx.shape[0] == 1000000, f"Expected 1M transactions, got {tx.shape[0]}"
    assert cust.shape[0] == 100000, f"Expected 100K customers, got {cust.shape[0]}"
    assert agents.shape[0] == 5000, f"Expected 5K agents, got {agents.shape[0]}"
    assert merchants.shape[0] == 10000, f"Expected 10K merchants, got {merchants.shape[0]}"
    print("  PASS: test_data_loading")


def test_data_quality():
    """Test data quality checks."""
    tx = pd.read_parquet(os.path.join(r"E:\projects\AI-Based-Mobile-Banking-Fraud-Detection-System", "data", "raw", "transactions.parquet"))
    assert tx.shape[0] == 1000000
    assert tx.duplicated().sum() == 0
    assert tx.isnull().sum().sum() == 0
    print("  PASS: test_data_quality")


def test_schema():
    """Test that expected columns exist."""
    tx = pd.read_parquet(os.path.join(r"E:\projects\AI-Based-Mobile-Banking-Fraud-Detection-System", "data", "raw", "transactions.parquet"))
    expected_cols = ["transaction_id", "timestamp", "amount_bdt", "is_fraud", "fraud_type"]
    for c in expected_cols:
        assert c in tx.columns, f"Missing column: {c}"
    print("  PASS: test_schema")


def test_fraud_distribution():
    """Test fraud distribution matches expected."""
    tx = pd.read_parquet(os.path.join(r"E:\projects\AI-Based-Mobile-Banking-Fraud-Detection-System", "data", "raw", "transactions.parquet"))
    fraud_count = int(tx["is_fraud"].sum())
    assert fraud_count == 50000, f"Expected 50K fraud, got {fraud_count}"
    print("  PASS: test_fraud_distribution")


if __name__ == "__main__":
    test_data_loading()
    test_data_quality()
    test_schema()
    test_fraud_distribution()
    print("\nAll data tests passed!")