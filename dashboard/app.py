#!/usr/bin/env python
"""Streamlit dashboard for the AI-MFS Fraud Detection system.

All figures and metrics are computed from the real processed datasets and the
real model-evaluation reports stored in this project. Nothing is fabricated:
when an artifact is unavailable the UI says so explicitly instead of guessing.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Paths & configuration
# --------------------------------------------------------------------------- #
# dashboard/app.py lives one level below the project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
PREPROCESSING_DIR = os.path.join(MODEL_DIR, "preprocessing")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
MODEL_RESULTS_DIR = os.path.join(REPORTS_DIR, "model_results")

# Optional import of the project's own risk-scoring helper (kept optional so the
# dashboard still runs if the explainability dependencies are missing).
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MAX_SCORE_ROWS = 5000  # cap model scoring for responsiveness
SPLITS = {"Train": "train", "Validation": "val", "Test": "test"}

COLOR_LEGIT = "#22c55e"
COLOR_FRAUD = "#ef4444"
COLOR_PRIMARY = "#3b82f6"
RISK_ORDER = ["Low", "Medium", "High", "Critical"]
RISK_COLORS = {
    "Low": "#22c55e",
    "Medium": "#f59e0b",
    "High": "#f97316",
    "Critical": "#ef4444",
    "Unavailable": "#6b7280",
}

st.set_page_config(
    page_title="AI-MFS Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading dataset...")
def load_split(split_key: str) -> pd.DataFrame:
    """Load a processed parquet split (train/val/test)."""
    path = os.path.join(PROCESSED_DIR, f"{split_key}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    return pd.read_parquet(path)


def _find_first(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts() -> dict:
    """Load the trained model, scaler and feature metadata (fail-soft)."""
    arts = {"model": None, "model_name": None, "scaler": None, "feature_info": None, "error": None}
    try:
        scaler_path = _find_first(
            [
                os.path.join(MODEL_DIR, "scaler.pkl"),
                os.path.join(PREPROCESSING_DIR, "scaler.pkl"),
            ]
        )
        if scaler_path:
            arts["scaler"] = joblib.load(scaler_path)

        info_path = _find_first(
            [
                os.path.join(PREPROCESSING_DIR, "feature_info.pkl"),
                os.path.join(MODEL_DIR, "feature_info.pkl"),
            ]
        )
        if info_path:
            arts["feature_info"] = joblib.load(info_path)

        preferred = [
            "lightgbm_model.pkl",
            "xgboost_model.pkl",
            "random_forest_model.pkl",
            "decision_tree_model.pkl",
            "logistic_regression_model.pkl",
        ]
        model_path = _find_first([os.path.join(MODEL_DIR, n) for n in preferred])
        if model_path is None and os.path.isdir(MODEL_DIR):
            for f in sorted(os.listdir(MODEL_DIR)):
                if f.endswith("_model.pkl") and "isolation" not in f and f != "scaler.pkl":
                    model_path = os.path.join(MODEL_DIR, f)
                    break
        if model_path:
            arts["model"] = joblib.load(model_path)
            arts["model_name"] = os.path.basename(model_path)
    except Exception as exc:  # pragma: no cover - defensive
        arts["error"] = str(exc)
    return arts


def _model_feature_columns(arts: dict) -> list:
    scaler = arts.get("scaler")
    names = getattr(scaler, "feature_names_in_", None)
    if names is not None:
        return list(names)
    info = arts.get("feature_info") or {}
    return list(info.get("numeric_cols", []))


@st.cache_data(show_spinner=False)
def score_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a model fraud probability to each row of ``df``.

    ``df`` must already contain the model feature columns. Rows whose features
    are missing get ``NaN`` probability (surfaced as 'Unavailable').
    """
    arts = load_artifacts()
    out = df.copy()
    out["fraud_probability"] = np.nan

    model = arts.get("model")
    scaler = arts.get("scaler")
    cols = _model_feature_columns(arts)
    if model is None or scaler is None or not cols:
        return out

    missing = [c for c in cols if c not in out.columns]
    if missing:
        return out

    try:
        X = out[cols].astype(float)
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
        X_scaled = scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=cols, index=out.index)
        proba = model.predict_proba(X_scaled)[:, 1]
        out["fraud_probability"] = np.clip(proba, 0.0, 1.0)
    except Exception:
        # Leave NaN so the UI reports probabilities as unavailable.
        pass
    return out


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    path = _find_first(
        [
            os.path.join(REPORTS_DIR, "model_comparison.csv"),
            os.path.join(MODEL_RESULTS_DIR, "model_comparison.csv"),
        ]
    )
    if not path:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_results_json() -> dict:
    path = _find_first(
        [
            os.path.join(REPORTS_DIR, "results.json"),
            os.path.join(MODEL_RESULTS_DIR, "results.json"),
        ]
    )
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_feature_importance(model_name: str) -> pd.DataFrame:
    if not model_name:
        return pd.DataFrame()
    slug = model_name.lower().replace(" ", "_")
    candidates = [
        os.path.join(MODEL_RESULTS_DIR, f"{slug}_feature_importance.csv"),
        os.path.join(REPORTS_DIR, f"{slug}_feature_importance.csv"),
    ]
    path = _find_first(candidates)
    if not path:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# Formatting / helper utilities
# --------------------------------------------------------------------------- #
def fmt_int(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def fmt_bdt(value) -> str:
    try:
        return f"{float(value):,.0f} BDT"
    except (TypeError, ValueError):
        return "—"


def risk_level_series(prob: pd.Series) -> pd.Series:
    bins = [-np.inf, 0.3, 0.7, 0.9, np.inf]
    labels = ["Low", "Medium", "High", "Critical"]
    cut = pd.cut(prob, bins=bins, labels=labels, right=False)
    return cut.astype("object").where(prob.notna(), "Unavailable").astype(str)


def add_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "fraud_probability" in out.columns:
        out["risk_level"] = risk_level_series(out["fraud_probability"])
    else:
        out["risk_level"] = "Unavailable"
    return out


def kpi(col, label: str, value: str, help_text: str | None = None):
    with col:
        with st.container(border=True):
            st.metric(label, value, help=help_text)


# --------------------------------------------------------------------------- #
# Page: Overview
# --------------------------------------------------------------------------- #
def render_overview(df: pd.DataFrame, split_label: str) -> None:
    st.title("Overview")
    st.caption(f"Dataset: {split_label} split • {fmt_int(len(df))} transactions")

    total = len(df)
    fraud = int(df["is_fraud"].sum()) if "is_fraud" in df.columns else 0
    rate = fraud / total if total else 0.0
    volume = float(df["amount_bdt"].sum()) if "amount_bdt" in df.columns else 0.0
    avg_amount = float(df["amount_bdt"].mean()) if total and "amount_bdt" in df.columns else 0.0

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Total transactions", fmt_int(total))
    kpi(c2, "Fraud transactions", fmt_int(fraud))
    kpi(c3, "Fraud rate", f"{rate:.2%}")
    kpi(c4, "Total amount", fmt_bdt(volume))

    c5, c6, c7, c8 = st.columns(4)
    avg_fraud = (
        float(df.loc[df["is_fraud"] == 1, "amount_bdt"].mean())
        if fraud and "amount_bdt" in df.columns
        else 0.0
    )
    kpi(c5, "Average amount", fmt_bdt(avg_amount))
    kpi(c6, "Average fraud amount", fmt_bdt(avg_fraud) if fraud else "—")
    kpi(c7, "Legitimate transactions", fmt_int(total - fraud))
    kpi(c8, "Unique transaction types", fmt_int(df["transaction_type"].nunique()) if "transaction_type" in df.columns else "—")

    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader(":material/donut_large: Fraud distribution")
        fig = px.pie(
            names=["Legitimate", "Fraud"],
            values=[total - fraud, fraud],
            hole=0.55,
            color=["Legitimate", "Fraud"],
            color_discrete_map={"Legitimate": COLOR_LEGIT, "Fraud": COLOR_FRAUD},
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, width="stretch", alt="Fraud versus legitimate transaction distribution")

    with col_right:
        st.subheader(":material/bar_chart: Transactions by type")
        if "transaction_type" in df.columns:
            type_stats = (
                df.groupby("transaction_type")
                .agg(transactions=("is_fraud", "size"), fraud=("is_fraud", "sum"))
                .reset_index()
                .sort_values("transactions", ascending=False)
            )
            fig = px.bar(
                type_stats,
                x="transaction_type",
                y="transactions",
                color="fraud",
                labels={"transaction_type": "Transaction type", "transactions": "Transactions", "fraud": "Fraud"},
                color_continuous_scale=[[0, COLOR_LEGIT], [1, COLOR_FRAUD]],
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), xaxis_tickangle=-30)
            st.plotly_chart(fig, width="stretch", alt="Transaction counts by type with fraud overlay")
        else:
            st.info("Transaction type column is not available in this dataset.")

    st.subheader(":material/trending_up: Transaction trend by hour")
    if "hour" in df.columns:
        hourly = (
            df.groupby("hour")
            .agg(transactions=("is_fraud", "size"), fraud=("is_fraud", "sum"))
            .reset_index()
            .sort_values("hour")
        )
        fig = go.Figure()
        fig.add_bar(
            x=hourly["hour"],
            y=hourly["transactions"],
            name="All transactions",
            marker_color=COLOR_PRIMARY,
            opacity=0.75,
        )
        fig.add_scatter(
            x=hourly["hour"],
            y=hourly["fraud"],
            name="Fraud",
            mode="lines+markers",
            line=dict(color=COLOR_FRAUD, width=3),
            yaxis="y2",
        )
        fig.update_layout(
            xaxis=dict(title="Hour of day", dtick=1),
            yaxis=dict(title="All transactions"),
            yaxis2=dict(title="Fraud", overlaying="y", side="right"),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=30, b=10, l=10, r=10),
        )
        st.plotly_chart(fig, width="stretch", alt="Hourly transaction volume with fraud overlay")
    else:
        st.info("Hour column is not available in this dataset.")

    st.subheader(":material/receipt_long: Recent transactions")
    recent_cols = [
        c
        for c in ["transaction_id", "timestamp", "amount_bdt", "transaction_type", "provider", "is_fraud"]
        if c in df.columns
    ]
    recent = df.sort_values("timestamp", ascending=False).head(10)[recent_cols].copy()
    if "is_fraud" in recent.columns:
        recent["status"] = np.where(recent["is_fraud"] == 1, "Fraud", "Legitimate")
    st.dataframe(
        recent,
        hide_index=True,
        column_config={
            "transaction_id": "Transaction ID",
            "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss"),
            "amount_bdt": st.column_config.NumberColumn("Amount (BDT)", format="%.2f"),
            "transaction_type": "Type",
            "provider": "Provider",
            "is_fraud": None,
            "status": "Status",
        },
    )


# --------------------------------------------------------------------------- #
# Page: Transaction Monitoring
# --------------------------------------------------------------------------- #
def render_monitoring(df: pd.DataFrame, split_label: str) -> None:
    st.title("Transaction Monitoring")
    st.caption(f"Dataset: {split_label} split • {fmt_int(len(df))} transactions available")

    with st.container(border=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            search = st.text_input("Search transaction ID", placeholder="e.g. TXN...")
        with f2:
            types = sorted(df["transaction_type"].dropna().unique().tolist()) if "transaction_type" in df.columns else []
            selected_types = st.multiselect("Transaction type", types, default=types)
        with f3:
            status_filter = st.selectbox("Status", ["All", "Fraud", "Legitimate"])

        f4, f5, f6 = st.columns(3)
        with f4:
            if "amount_bdt" in df.columns and len(df):
                amin = float(np.floor(df["amount_bdt"].min()))
                amax = float(np.ceil(df["amount_bdt"].max()))
            else:
                amin, amax = 0.0, 1.0
            if amax <= amin:
                amax = amin + 1.0
            amount_range = st.slider("Amount range (BDT)", min_value=amin, max_value=amax, value=(amin, amax))
        with f5:
            if "timestamp" in df.columns:
                dmin = df["timestamp"].dt.date.min()
                dmax = df["timestamp"].dt.date.max()
            else:
                dmin = pd.Timestamp("1970-01-01").date()
                dmax = pd.Timestamp("1970-01-02").date()
            date_value = st.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
            if isinstance(date_value, (tuple, list)):
                date_start, date_end = (date_value[0], date_value[-1]) if len(date_value) else (dmin, dmax)
            else:
                date_start = date_end = date_value
        with f6:
            hour_range = st.slider("Hour of day", min_value=0, max_value=23, value=(0, 23)) if "hour" in df.columns else (0, 23)

    filtered = df
    if search and "transaction_id" in filtered.columns:
        filtered = filtered[filtered["transaction_id"].astype(str).str.contains(search, case=False, na=False)]
    if selected_types and "transaction_type" in filtered.columns:
        filtered = filtered[filtered["transaction_type"].isin(selected_types)]
    if "amount_bdt" in filtered.columns:
        filtered = filtered[(filtered["amount_bdt"] >= amount_range[0]) & (filtered["amount_bdt"] <= amount_range[1])]
    if "timestamp" in filtered.columns:
        mask = (filtered["timestamp"].dt.date >= date_start) & (filtered["timestamp"].dt.date <= date_end)
        filtered = filtered[mask]
    if "hour" in filtered.columns:
        filtered = filtered[(filtered["hour"] >= hour_range[0]) & (filtered["hour"] <= hour_range[1])]
    if status_filter != "All" and "is_fraud" in filtered.columns:
        target = 1 if status_filter == "Fraud" else 0
        filtered = filtered[filtered["is_fraud"] == target]

    matching = len(filtered)
    fraud_matching = int(filtered["is_fraud"].sum()) if "is_fraud" in filtered.columns and matching else 0
    avg_matching = float(filtered["amount_bdt"].mean()) if matching and "amount_bdt" in filtered.columns else 0.0

    m1, m2, m3 = st.columns(3)
    kpi(m1, "Matching transactions", fmt_int(matching))
    kpi(m2, "Fraud among matches", fmt_int(fraud_matching))
    kpi(m3, "Average amount", fmt_bdt(avg_matching))

    if matching == 0:
        st.warning("No transactions match the current filters. Adjust the filters and try again.")
        return

    display_cols = [
        c
        for c in [
            "transaction_id",
            "timestamp",
            "amount_bdt",
            "transaction_type",
            "provider",
            "hour",
            "is_fraud",
        ]
        if c in filtered.columns
    ]
    sample = (
        filtered.sort_values("timestamp", ascending=False) if "timestamp" in filtered.columns else filtered
    ).head(MAX_SCORE_ROWS).copy()

    with st.spinner("Scoring transactions with the trained model..."):
        scored_full = score_transactions(sample.copy()) if _artifacts_ready() else sample.copy()
        scored_full = add_risk_level(scored_full)
    keep_cols = display_cols + [c for c in ["fraud_probability", "risk_level"] if c in scored_full.columns]
    scored = scored_full[keep_cols].copy()

    if _artifacts_ready():
        st.caption(
            f"Showing the {min(matching, MAX_SCORE_ROWS):,} most recent matching transactions "
            f"(of {matching:,}) with model-scored fraud probability."
        )
    else:
        st.info(
            "Model artifacts are unavailable, so fraud probability is not shown. "
            "The transaction status column still reflects the dataset's `is_fraud` label.",
            icon=":material/info:",
        )

    if "is_fraud" in scored.columns:
        scored["status"] = np.where(scored["is_fraud"] == 1, "Fraud", "Legitimate")

    st.dataframe(
        scored,
        hide_index=True,
        column_config={
            "transaction_id": "Transaction ID",
            "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss"),
            "amount_bdt": st.column_config.NumberColumn("Amount (BDT)", format="%.2f"),
            "transaction_type": "Type",
            "provider": "Provider",
            "hour": st.column_config.NumberColumn("Hour", format="%d"),
            "is_fraud": None,
            "fraud_probability": st.column_config.ProgressColumn(
                "Fraud probability", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "risk_level": "Risk level",
            "status": "Status",
        },
    )


# --------------------------------------------------------------------------- #
# Page: Fraud Alerts
# --------------------------------------------------------------------------- #
def render_alerts(df: pd.DataFrame, split_label: str) -> None:
    st.title("Fraud Alerts")
    st.caption(f"Dataset: {split_label} split • confirmed fraud labels from the dataset")

    if "is_fraud" not in df.columns:
        st.error("The dataset does not contain an `is_fraud` label, so alerts cannot be derived.")
        return

    fraud_df = df[df["is_fraud"] == 1].copy()
    total = len(df)
    fraud_count = len(fraud_df)
    fraud_rate = fraud_count / total if total else 0.0
    avg_fraud = float(fraud_df["amount_bdt"].mean()) if fraud_count and "amount_bdt" in fraud_df.columns else 0.0

    if fraud_count == 0:
        st.success("No fraudulent transactions were found in this dataset split.")
        return

    display_cols = [
        c
        for c in [
            "transaction_id",
            "timestamp",
            "amount_bdt",
            "transaction_type",
            "provider",
            "fraud_type",
            "hour",
        ]
        if c in fraud_df.columns
    ]
    sample = (
        fraud_df.sort_values("timestamp", ascending=False) if "timestamp" in fraud_df.columns else fraud_df
    ).head(MAX_SCORE_ROWS).copy()

    with st.spinner("Scoring fraud alerts with the trained model..."):
        scored_full = score_transactions(sample.copy()) if _artifacts_ready() else sample.copy()
        scored_full = add_risk_level(scored_full)
    keep_cols = display_cols + [c for c in ["fraud_probability", "risk_level"] if c in scored_full.columns]
    scored = scored_full[keep_cols].copy()

    high_critical = int(scored["risk_level"].isin(["High", "Critical"]).sum()) if _artifacts_ready() else 0

    a1, a2, a3, a4 = st.columns(4)
    kpi(a1, "Confirmed fraud alerts", fmt_int(fraud_count))
    kpi(a2, "Fraud rate", f"{fraud_rate:.2%}")
    kpi(a3, "High / critical (scored sample)", fmt_int(high_critical) if _artifacts_ready() else "—")
    kpi(a4, "Average fraud amount", fmt_bdt(avg_fraud))

    if not _artifacts_ready():
        st.info(
            "Model artifacts are unavailable, so risk levels and fraud probabilities cannot be computed. "
            "Alerts below are the dataset's confirmed fraud transactions.",
            icon=":material/info:",
        )

    st.divider()
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader(":material/warning: Risk level distribution")
        if _artifacts_ready():
            counts = scored["risk_level"].value_counts().reindex(RISK_ORDER, fill_value=0)
            fig = px.bar(
                x=counts.index,
                y=counts.values,
                labels={"x": "Risk level", "y": "Transactions"},
                color=counts.index,
                color_discrete_map=RISK_COLORS,
            )
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, width="stretch", alt="Fraud alert counts by model risk level")
            st.caption(f"Risk levels computed by the model on {len(scored):,} scored fraud transactions.")
        else:
            st.info("Risk levels require the trained model artifacts.")
    with col_right:
        st.subheader(":material/category: Fraud type breakdown")
        if "fraud_type" in fraud_df.columns:
            type_counts = fraud_df["fraud_type"].replace("none", "unknown").value_counts().reset_index()
            type_counts.columns = ["fraud_type", "count"]
            fig = px.bar(
                type_counts,
                x="count",
                y="fraud_type",
                orientation="h",
                labels={"count": "Transactions", "fraud_type": "Fraud type"},
                color_discrete_sequence=[COLOR_FRAUD],
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch", alt="Confirmed fraud transactions by fraud type")
        else:
            st.info("Fraud type column is not available.")

    st.subheader(":material/list_alt: Fraud alert details")
    f1, f2, f3 = st.columns(3)
    with f1:
        if _artifacts_ready():
            level_options = [lvl for lvl in RISK_ORDER if lvl in scored["risk_level"].unique()]
            selected_levels = st.multiselect("Risk level", level_options, default=level_options)
        else:
            selected_levels = []
    with f2:
        if "fraud_type" in scored.columns:
            ftypes = sorted(scored["fraud_type"].dropna().unique().tolist())
            selected_ftypes = st.multiselect("Fraud type", ftypes, default=ftypes)
        else:
            selected_ftypes = []
    with f3:
        min_prob = st.slider("Minimum fraud probability", 0.0, 1.0, 0.0, 0.05) if _artifacts_ready() else 0.0

    table = scored
    if selected_levels and "risk_level" in table.columns:
        table = table[table["risk_level"].isin(selected_levels)]
    if selected_ftypes and "fraud_type" in table.columns:
        table = table[table["fraud_type"].isin(selected_ftypes)]
    if _artifacts_ready() and "fraud_probability" in table.columns:
        table = table[table["fraud_probability"].fillna(0.0) >= min_prob]

    if table.empty:
        st.warning("No fraud alerts match the current filters.")
        return

    if "is_fraud" in table.columns:
        table = table.copy()
        table["status"] = "Fraud"

    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "transaction_id": "Transaction ID",
            "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss"),
            "amount_bdt": st.column_config.NumberColumn("Amount (BDT)", format="%.2f"),
            "transaction_type": "Type",
            "provider": "Provider",
            "fraud_type": "Fraud type",
            "hour": st.column_config.NumberColumn("Hour", format="%d"),
            "fraud_probability": st.column_config.ProgressColumn(
                "Fraud probability", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "risk_level": "Risk level",
            "status": "Status",
        },
    )


# --------------------------------------------------------------------------- #
# Page: Model Performance
# --------------------------------------------------------------------------- #
def render_model_performance() -> None:
    st.title("Model Performance")
    comparison = load_model_comparison()
    results = load_results_json()

    if comparison.empty and not results:
        st.warning(
            "No model evaluation reports were found. Run `python -m src.models.train` to generate "
            "`reports/model_comparison.csv` and `reports/results.json`."
        )
        return

    model_names = list(results.keys()) if results else comparison["model"].tolist()
    default_idx = 0
    if "LightGBM" in model_names:
        default_idx = model_names.index("LightGBM")
    model_name = st.selectbox("Model", model_names, index=default_idx)

    metrics = (results.get(model_name) or {}).get("test_metrics", {})
    if not metrics and not comparison.empty:
        row = comparison[comparison["model"] == model_name]
        if not row.empty:
            r = row.iloc[0]
            metrics = {
                "precision": r.get("precision", np.nan),
                "recall": r.get("recall", np.nan),
                "f1": r.get("f1", np.nan),
                "roc_auc": r.get("roc_auc", np.nan),
                "pr_auc": r.get("pr_auc", np.nan),
            }

    if not metrics:
        st.error(f"No evaluation metrics are available for {model_name}.")
        return

    tn = int(metrics.get("tn", 0) or 0)
    fp = int(metrics.get("fp", 0) or 0)
    fn = int(metrics.get("fn", 0) or 0)
    tp = int(metrics.get("tp", 0) or 0)
    support = tn + fp + fn + tp
    accuracy = (tp + tn) / support if support else np.nan

    st.caption("Metrics are computed on the held-out test split and read from `reports/results.json`.")

    c1, c2, c3 = st.columns(3)
    kpi(c1, "Accuracy", f"{accuracy:.4f}" if not np.isnan(accuracy) else "—", "Derived from the test confusion matrix")
    kpi(c2, "Precision", f"{float(metrics.get('precision', np.nan)):.4f}")
    kpi(c3, "Recall", f"{float(metrics.get('recall', np.nan)):.4f}")
    c4, c5, c6 = st.columns(3)
    kpi(c4, "F1 score", f"{float(metrics.get('f1', np.nan)):.4f}")
    roc_auc = float(metrics.get("roc_auc", 0.0) or 0.0)
    kpi(c5, "ROC-AUC", f"{roc_auc:.4f}" if roc_auc > 0 else "N/A")
    pr_auc = float(metrics.get("pr_auc", 0.0) or 0.0)
    kpi(c6, "PR-AUC", f"{pr_auc:.4f}" if pr_auc > 0 else "N/A")

    st.divider()
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader(":material/grid_on: Confusion matrix")
        if support:
            z = [[tn, fp], [fn, tp]]
            fig = go.Figure(
                go.Heatmap(
                    z=z,
                    x=["Predicted legitimate", "Predicted fraud"],
                    y=["Actual legitimate", "Actual fraud"],
                    text=[[fmt_int(v) for v in row] for row in z],
                    texttemplate="%{text}",
                    colorscale="Blues",
                    showscale=False,
                )
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, width="stretch", alt="Confusion matrix for the selected model")
            st.caption(
                f"TP={fmt_int(tp)} • FP={fmt_int(fp)} • FN={fmt_int(fn)} • TN={fmt_int(tn)}"
            )
        else:
            st.info("Confusion matrix values are not available for this model (unsupervised).")

    with col_right:
        st.subheader(":material/leaderboard: Model comparison")
        if not comparison.empty and {"model", "f1"}.issubset(comparison.columns):
            metric_cols = [c for c in ["precision", "recall", "f1", "roc_auc", "pr_auc"] if c in comparison.columns]
            melted = comparison.melt(id_vars="model", value_vars=metric_cols, var_name="metric", value_name="score")
            melted = melted[melted["score"] > 0]  # drop N/A (0) unsupervised rows
            fig = px.bar(
                melted,
                x="model",
                y="score",
                color="metric",
                barmode="group",
                labels={"model": "Model", "score": "Score", "metric": "Metric"},
            )
            fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis_tickangle=-30,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, width="stretch", alt="Comparison of model metrics across all trained models")
            best = comparison.loc[comparison["f1"].idxmax()]
            st.success(f"Best F1: {best['model']} ({float(best['f1']):.4f})")
        else:
            st.info("`reports/model_comparison.csv` is not available.")

    st.subheader(":material/insights: Feature importance")
    importance = load_feature_importance(model_name)
    if importance.empty:
        st.info(f"Feature importance is not available for {model_name} in the reports directory.")
    else:
        top = importance.head(15).iloc[::-1]
        fig = px.bar(
            top,
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Importance", "feature": "Feature"},
            color_discrete_sequence=[COLOR_PRIMARY],
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, width="stretch", alt="Top feature importances for the selected model")


# --------------------------------------------------------------------------- #
# App shell
# --------------------------------------------------------------------------- #
def _artifacts_ready() -> bool:
    arts = load_artifacts()
    return arts.get("model") is not None and arts.get("scaler") is not None and bool(_model_feature_columns(arts))


def main() -> None:
    st.sidebar.title(":material/shield: AI-MFS Fraud Detection")
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Transaction Monitoring", "Fraud Alerts", "Model Performance"],
    )

    split_label = st.sidebar.selectbox("Dataset split", list(SPLITS.keys()), index=0)
    split_key = SPLITS[split_label]

    arts = load_artifacts()
    with st.sidebar:
        st.divider()
        if _artifacts_ready():
            st.success(f"Model: {arts.get('model_name')}", icon=":material/check_circle:")
        else:
            st.warning("Model artifacts unavailable", icon=":material/warning:")
            if arts.get("error"):
                st.caption(f"Details: {arts['error']}")
        st.caption("Synthetic academic dataset • Bangladesh MFS ecosystem")

    if page == "Model Performance":
        render_model_performance()
        return

    try:
        with st.spinner(f"Loading {split_label.lower()} dataset..."):
            df = load_split(split_key)
    except FileNotFoundError as exc:
        st.error(f"Dataset unavailable: {exc}")
        return
    except Exception as exc:  # pragma: no cover - defensive
        st.exception(exc)
        return

    if df.empty:
        st.warning("The selected dataset split is empty.")
        return

    if page == "Overview":
        render_overview(df, split_label)
    elif page == "Transaction Monitoring":
        render_monitoring(df, split_label)
    elif page == "Fraud Alerts":
        render_alerts(df, split_label)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # surface errors instead of a silent blank screen
        st.error("The dashboard encountered an unexpected error.")
        st.exception(exc)
