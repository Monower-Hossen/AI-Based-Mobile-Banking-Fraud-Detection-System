#!/usr/bin/env python
# Streamlit Dashboard for AI-MFS-Fraud-Detection ( simplified version )
import json
import os
import sys
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="AI-MFS-Fraud-Detection", layout="wide", initial_sidebar_state="expanded")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

st.sidebar.title("AI-MFS-Fraud-Detection")
page = st.sidebar.radio("Navigation", ["Overview", "Transaction Monitoring", "Fraud Alerts", "Model Performance"])

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

@st.cache_resource
def load_artifacts():
    arts = {"models": {}, "scaler": None, "feature_info": None}
    if os.path.exists(os.path.join(MODEL_DIR, "scaler.pkl")):
        arts["scaler"] = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    if os.path.exists(os.path.join(MODEL_DIR, "feature_info.pkl")):
        arts["feature_info"] = joblib.load(os.path.join(MODEL_DIR, "feature_info.pkl"))
    if os.path.exists(MODEL_DIR):
        for f in os.listdir(MODEL_DIR):
            if f.endswith("_model.pkl") and f != "scaler.pkl":
                try:
                    arts["models"][f] = joblib.load(os.path.join(MODEL_DIR, f))
                except Exception:
                    pass
    return arts

@st.cache_data
def load_data():
    data = {}
    for name in ["train", "val", "test"]:
        p = os.path.join(PROCESSED_DIR, f"{name}.parquet")
        if os.path.exists(p):
            data[name] = pd.read_parquet(p)
    return data

arts = load_artifacts()
data = load_data()

if st.sidebar.radio("Go to", ["Overview", "Model Performance"]) == "Overview":
    tx = data.get("train", pd.DataFrame())
    if not tx.empty:
        st.title("Overview")
        c1, c2, c3, c4 = st.columns(4)
        tc = len(tx)
        fc = int(tx["is_fraud"].sum()) if "is_fraud" in tx.columns else 0
        fr = fc / tc if tc > 0 else 0
        tv = float(tx["amount_bdt"].sum()) if "amount_bdt" in tx.columns else 0
        st.metric("Total Transactions", tc)
        st.metric("Fraud Transactions", fc)
        st.metric("Fraud Rate", "{:.2%}".format(fr))
        st.metric("Total Volume (BDT)", "{:,.0f}".format(tv))
        if "transaction_type" in tx.columns:
            px.pie(tx, names="transaction_type", values="amount_bdt", hole=0.4).show()
        if "timestamp" in tx.columns:
            tx_s = tx.sort_values("timestamp")
            px.line(tx_s.head(100), x="timestamp", y="amount_bdt", color="is_fraud").show()
else:
    st.title("Model Performance")
    cmp_path = os.path.join(PROJECT_ROOT, "reports", "model_results", "model_comparison.csv")
    if os.path.exists(cmp_path):
        import pandas as pd
        cd = pd.read_csv(cmp_path)
        st.dataframe(cd)
        if not cd.empty:
            best = cd.loc[cd["f1"].idxmax()]
            st.success("Best: {} with F1 {:.4f}".format(best["model"], best["f1"]))