#!/usr/bin/env python
"""Streamlit dashboard for the AI-MFS Fraud Detection system.

All figures and metrics are computed from the real processed datasets and the
real model-evaluation reports stored in this project. Nothing is fabricated:
when an artifact is unavailable the UI says so explicitly instead of guessing.
"""
import importlib.util
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
MAX_EXPORT_ROWS = 50000
MAX_BATCH_ROWS = 1000
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
AMOUNT_BINS = [0, 1000, 5000, 10000, 50000, 100000, np.inf]
AMOUNT_LABELS = ["0–1k", "1k–5k", "5k–10k", "10k–50k", "50k–100k", "100k+"]

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


def _artifacts_ready() -> bool:
    arts = load_artifacts()
    return arts.get("model") is not None and arts.get("scaler") is not None and bool(_model_feature_columns(arts))


def current_split():
    label = st.session_state.get("split_label", "Train")
    try:
        return load_split(SPLITS[label]), label
    except FileNotFoundError as exc:
        st.error(f"Dataset unavailable: {exc}")
        st.stop()


def score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return add_risk_level(frame.copy())
    if not _artifacts_ready():
        return add_risk_level(frame.copy())
    with st.spinner("Scoring transactions with the trained model..."):
        return add_risk_level(score_transactions(frame.copy()))


def risk_info(probability):
    if probability is None:
        return None
    try:
        value = float(probability)
    except (TypeError, ValueError):
        return None
    if np.isnan(value):
        return None
    try:
        from src.explainability.shap_explainer import compute_risk_score

        return compute_risk_score(value)
    except Exception:
        return None


def shap_explanation_for_row(row: pd.Series, reference: pd.DataFrame):
    arts = load_artifacts()
    cols = _model_feature_columns(arts)
    model, scaler = arts.get("model"), arts.get("scaler")
    if model is None or scaler is None or not cols:
        return None
    try:
        values = pd.to_numeric(row.reindex(cols), errors="coerce")
        medians = reference.reindex(columns=cols).median(numeric_only=True).fillna(0.0)
        X = pd.DataFrame([values.fillna(medians).fillna(0.0)], columns=cols).astype(float)
        X_scaled = pd.DataFrame(scaler.transform(X), columns=cols)
        from src.explainability.shap_explainer import generate_local_explanation

        return generate_local_explanation(model, X_scaled, cols, 0)
    except Exception:
        return None


def shap_global_importance(sample: pd.DataFrame):
    arts = load_artifacts()
    cols = _model_feature_columns(arts)
    model, scaler = arts.get("model"), arts.get("scaler")
    if model is None or scaler is None or not cols:
        return pd.DataFrame()
    try:
        X = sample.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
        X_scaled = pd.DataFrame(scaler.transform(X), columns=cols)
        from src.explainability.shap_explainer import generate_global_importance

        return generate_global_importance(model, X_scaled, cols)
    except Exception:
        return pd.DataFrame()


def date_bounds(df: pd.DataFrame):
    if "timestamp" not in df.columns or df.empty:
        return None, None
    return df["timestamp"].dt.date.min(), df["timestamp"].dt.date.max()


def apply_date_filter(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    if "timestamp" not in df.columns or df.empty:
        return df
    dmin, dmax = date_bounds(df)
    picked = st.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax, key=f"{key_prefix}_date_range")
    if isinstance(picked, (tuple, list)):
        if not picked:
            return df
        start, end = picked[0], picked[-1]
    else:
        start = end = picked
    return df[(df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)]


def type_filter(df: pd.DataFrame, key_prefix: str) -> list:
    if "transaction_type" not in df.columns:
        return []
    options = sorted(df["transaction_type"].dropna().unique().tolist())
    return st.multiselect("Transaction type", options, default=options, key=f"{key_prefix}_types")


def csv_download(frame: pd.DataFrame, label: str, filename: str, key: str) -> None:
    payload = frame.head(MAX_EXPORT_ROWS).to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=label, data=payload, file_name=filename, mime="text/csv", key=key, width="stretch")
    if len(frame) > MAX_EXPORT_ROWS:
        st.caption(f"Export capped at {MAX_EXPORT_ROWS:,} of {len(frame):,} rows.")


def style_fig(fig, height: int = 380):
    fig.update_layout(
        margin=dict(t=30, b=10, l=10, r=10),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def transaction_column_config() -> dict:
    return {
        "transaction_id": "Transaction ID",
        "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss"),
        "amount_bdt": st.column_config.NumberColumn("Amount (BDT)", format="%.2f"),
        "transaction_type": "Type",
        "provider": "Provider",
        "sender_id": "Sender ID",
        "receiver_id": "Receiver ID",
        "sender_division": "Sender division",
        "sender_district": "Sender district",
        "receiver_division": "Receiver division",
        "receiver_district": "Receiver district",
        "fraud_type": "Fraud type",
        "hour": st.column_config.NumberColumn("Hour", format="%d"),
        "risk_level": "Risk level",
        "status": "Status",
        "fraud_probability": st.column_config.ProgressColumn(
            "Fraud probability", min_value=0.0, max_value=1.0, format="%.3f"
        ),
        "risk_score": st.column_config.NumberColumn("Risk score", format="%d"),
    }


def configured_columns(frame: pd.DataFrame, config: dict | None = None) -> dict:
    base = transaction_column_config()
    if config:
        base.update(config)
    return {key: value for key, value in base.items() if key in frame.columns}


def add_status(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "is_fraud" in out.columns:
        out["status"] = np.where(out["is_fraud"] == 1, "Fraud", "Legitimate")
    return out


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
    fraud_volume = (
        float(df.loc[df["is_fraud"] == 1, "amount_bdt"].sum())
        if fraud and "amount_bdt" in df.columns
        else 0.0
    )
    avg_amount = float(df["amount_bdt"].mean()) if total and "amount_bdt" in df.columns else 0.0

    sample = (
        df.sort_values("timestamp", ascending=False).head(MAX_SCORE_ROWS).copy()
        if "timestamp" in df.columns
        else df.head(MAX_SCORE_ROWS).copy()
    )
    scored = score_frame(sample)

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Total transactions", fmt_int(total))
    kpi(c2, "Fraud transactions", fmt_int(fraud))
    kpi(c3, "Legitimate transactions", fmt_int(total - fraud))
    kpi(c4, "Fraud rate", f"{rate:.2%}")

    c5, c6, c7, c8 = st.columns(4)
    kpi(c5, "Total fraud amount", fmt_bdt(fraud_volume))
    kpi(c6, "Total amount", fmt_bdt(volume))
    high_risk = int(scored["risk_level"].isin(["High", "Critical"]).sum()) if _artifacts_ready() else None
    kpi(c7, "High-risk transactions", fmt_int(high_risk) if high_risk is not None else "—")
    kpi(
        c8,
        "Unique transaction types",
        fmt_int(df["transaction_type"].nunique()) if "transaction_type" in df.columns else "—",
    )
    if _artifacts_ready():
        st.caption(f"High-risk count is model-scored on the {fmt_int(len(scored))} most recent transactions.")

    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader(":material/donut_large: Fraud vs legitimate")
        fig = px.pie(
            names=["Legitimate", "Fraud"],
            values=[total - fraud, fraud],
            hole=0.55,
            color=["Legitimate", "Fraud"],
            color_discrete_map={"Legitimate": COLOR_LEGIT, "Fraud": COLOR_FRAUD},
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig, width="stretch", alt="Fraud versus legitimate transaction distribution")

    with col_right:
        st.subheader(":material/bar_chart: Transaction type distribution")
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
            st.plotly_chart(
                style_fig(fig),
                width="stretch",
                alt="Transaction counts by type with fraud overlay",
            )
        else:
            st.info("Transaction type column is not available in this dataset.")

    st.subheader(":material/calendar_month: Daily fraud trends")
    if "timestamp" in df.columns:
        agg = {"transactions": ("is_fraud", "size"), "fraud": ("is_fraud", "sum")}
        if "amount_bdt" in df.columns:
            agg["fraud_amount"] = ("amount_bdt", "sum")
        daily = df.groupby(df["timestamp"].dt.date).agg(**agg).reset_index()
        daily = daily.rename(columns={"timestamp": "date"}).sort_values("date")
        fig = go.Figure()
        fig.add_bar(
            x=daily["date"],
            y=daily["transactions"],
            name="All transactions",
            marker_color=COLOR_PRIMARY,
            opacity=0.6,
        )
        fig.add_scatter(
            x=daily["date"],
            y=daily["fraud"],
            name="Fraud",
            mode="lines+markers",
            line=dict(color=COLOR_FRAUD, width=3),
            yaxis="y2",
        )
        fig.update_layout(
            xaxis=dict(title="Date"),
            yaxis=dict(title="All transactions"),
            yaxis2=dict(title="Fraud", overlaying="y", side="right"),
            hovermode="x unified",
        )
        st.plotly_chart(style_fig(fig, 340), width="stretch", alt="Daily transaction volume with fraud overlay")
    else:
        st.info("Timestamp column is not available in this dataset.")

    st.subheader(":material/schedule: Transaction trend by hour")
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
        )
        st.plotly_chart(style_fig(fig, 340), width="stretch", alt="Hourly transaction volume with fraud overlay")
    else:
        st.info("Hour column is not available in this dataset.")

    st.subheader(":material/receipt_long: Recent suspicious transactions")
    suspicious = scored.copy()
    mask = suspicious["is_fraud"].eq(1) if "is_fraud" in suspicious.columns else pd.Series(False, index=suspicious.index)
    if _artifacts_ready():
        mask = mask | suspicious["risk_level"].isin(["High", "Critical"])
    suspicious = suspicious[mask]
    if "timestamp" in suspicious.columns:
        suspicious = suspicious.sort_values("timestamp", ascending=False)
    suspicious = add_status(suspicious)
    show_cols = [
        c
        for c in [
            "transaction_id",
            "timestamp",
            "amount_bdt",
            "transaction_type",
            "provider",
            "fraud_probability",
            "risk_level",
            "status",
        ]
        if c in suspicious.columns
    ]
    if suspicious.empty:
        st.info("No suspicious transactions were found in the scored sample.")
    else:
        st.dataframe(
            suspicious.head(10)[show_cols],
            hide_index=True,
            column_config=configured_columns(suspicious),
            alt="Most recent suspicious transactions",
        )
    avg_fraud = (
        float(df.loc[df["is_fraud"] == 1, "amount_bdt"].mean())
        if fraud and "amount_bdt" in df.columns
        else 0.0
    )
    st.caption(f"Average amount: {fmt_bdt(avg_amount)} • Average fraud amount: {fmt_bdt(avg_fraud) if fraud else '—'}")


# --------------------------------------------------------------------------- #
# Page: Transactions
# --------------------------------------------------------------------------- #
def render_transactions(df: pd.DataFrame, split_label: str) -> None:
    st.title("Transactions")
    st.caption(f"Dataset: {split_label} split • {fmt_int(len(df))} transactions available")

    with st.container(border=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            search = st.text_input("Search", placeholder="Partial transaction ID", key="tx_search")
        with f2:
            tx_id = st.text_input("Transaction ID (exact)", placeholder="TXN...", key="tx_id_exact")
        with f3:
            customer_id = st.text_input("Customer ID (sender or receiver)", key="tx_customer_id")

        f4, f5, f6 = st.columns(3)
        with f4:
            selected_types = type_filter(df, "tx")
        with f5:
            status_filter = st.selectbox("Fraud status", ["All", "Fraud", "Legitimate"], key="tx_status")
        with f6:
            hour_range = (
                st.slider("Hour of day", min_value=0, max_value=23, value=(0, 23), key="tx_hours")
                if "hour" in df.columns
                else (0, 23)
            )

        f7, f8, f9 = st.columns(3)
        with f7:
            if "amount_bdt" in df.columns and len(df):
                amin = float(np.floor(df["amount_bdt"].min()))
                amax = float(np.ceil(df["amount_bdt"].max()))
                if amax <= amin:
                    amax = amin + 1.0
                amount_range = st.slider(
                    "Amount range (BDT)", min_value=amin, max_value=amax, value=(amin, amax), key="tx_amount"
                )
            else:
                amount_range = None
        with f8:
            apply_date = st.toggle("Filter by date range", value=False, key="tx_use_dates")
        with f9:
            st.caption("Customer IDs are stored as sender_id / receiver_id in this dataset.")

        if apply_date:
            date_picked = st.date_input("Date range", value=date_bounds(df), key="tx_dates")
            if isinstance(date_picked, (tuple, list)) and len(date_picked) == 2:
                date_start, date_end = date_picked
            elif isinstance(date_picked, (tuple, list)) and date_picked:
                date_start = date_end = date_picked[0]
            else:
                date_start = date_end = date_picked

    filtered = df
    if search and "transaction_id" in filtered.columns:
        filtered = filtered[filtered["transaction_id"].astype(str).str.contains(search, case=False, na=False)]
    if tx_id and "transaction_id" in filtered.columns:
        filtered = filtered[filtered["transaction_id"].astype(str) == tx_id.strip()]
    if customer_id:
        id_cols = [c for c in ["sender_id", "receiver_id", "customer_id"] if c in filtered.columns]
        if id_cols:
            mask = pd.Series(False, index=filtered.index)
            for col in id_cols:
                mask = mask | filtered[col].astype(str).str.contains(customer_id.strip(), case=False, na=False)
            filtered = filtered[mask]
    if selected_types and "transaction_type" in filtered.columns:
        filtered = filtered[filtered["transaction_type"].isin(selected_types)]
    if amount_range is not None and "amount_bdt" in filtered.columns:
        filtered = filtered[
            (filtered["amount_bdt"] >= amount_range[0]) & (filtered["amount_bdt"] <= amount_range[1])
        ]
    if apply_date and "timestamp" in filtered.columns and date_start is not None:
        filtered = filtered[
            (filtered["timestamp"].dt.date >= date_start) & (filtered["timestamp"].dt.date <= date_end)
        ]
    if "hour" in filtered.columns:
        filtered = filtered[(filtered["hour"] >= hour_range[0]) & (filtered["hour"] <= hour_range[1])]
    if status_filter != "All" and "is_fraud" in filtered.columns:
        filtered = filtered[filtered["is_fraud"] == (1 if status_filter == "Fraud" else 0)]

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
            "sender_id",
            "receiver_id",
            "hour",
            "is_fraud",
        ]
        if c in filtered.columns
    ]
    sort_col = st.selectbox("Sort by", display_cols, key="tx_sort_col")
    ascending = st.toggle("Ascending order", value=False, key="tx_sort_asc")
    filtered = filtered.sort_values(sort_col, ascending=ascending, kind="mergesort", na_position="last")

    page_size = st.selectbox("Rows per page", [25, 50, 100, 200], key="tx_page_size")
    total_pages = max(1, (matching - 1) // page_size + 1)
    if st.session_state.get("tx_page_no", 1) > total_pages:
        st.session_state["tx_page_no"] = total_pages
    page_no = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key="tx_page_no")
    start = (page_no - 1) * page_size
    page_df = filtered.iloc[start : start + page_size].copy()

    st.caption(f"Showing rows {start + 1:,}–{min(start + page_size, matching):,} of {matching:,}.")
    scored = score_frame(page_df)
    scored = add_status(scored)
    keep_cols = display_cols + [c for c in ["fraud_probability", "risk_level"] if c in scored.columns]
    keep_cols = list(dict.fromkeys(keep_cols))
    st.dataframe(
        scored[keep_cols],
        hide_index=True,
        column_config=configured_columns(scored),
        alt="Filtered transactions with model fraud probability",
    )
    if not _artifacts_ready():
        st.info(
            "Model artifacts are unavailable, so fraud probability is not shown. "
            "The status column still reflects the dataset's is_fraud label.",
            icon=":material/info:",
        )

    with st.expander("Transaction details"):
        detail_id = st.text_input("Transaction ID", key="tx_detail_id")
        if detail_id:
            detail = df[df["transaction_id"].astype(str) == detail_id.strip()]
            if detail.empty:
                st.warning("No transaction with that ID exists in the selected split.")
            else:
                row = detail.iloc[0]
                st.table(
                    pd.DataFrame({"Field": detail.columns, "Value": [str(row[c]) for c in detail.columns]}),
                    alt="All stored fields for the selected transaction",
                )
        else:
            st.caption("Enter a transaction ID above to inspect every stored field.")

    st.divider()
    st.subheader(":material/download: CSV export")
    export = add_status(filtered.head(MAX_EXPORT_ROWS))
    csv_download(export, "Download filtered transactions (CSV)", f"transactions_{SPLITS[split_label]}.csv", "tx_export_csv")


# --------------------------------------------------------------------------- #
# Page: Fraud detection
# --------------------------------------------------------------------------- #
def render_fraud_detection(df: pd.DataFrame, split_label: str) -> None:
    st.title("Fraud detection")
    st.caption("Interfaces for the project's existing model-scoring, risk-scoring and SHAP helpers")

    if not _artifacts_ready():
        st.error("Model artifacts are unavailable. Predictions cannot be produced.")
        return

    arts = load_artifacts()
    feature_cols = _model_feature_columns(arts)
    medians = df.reindex(columns=feature_cols).median(numeric_only=True).fillna(0.0)

    tab_single, tab_batch = st.tabs(["Single transaction", "Batch CSV"])

    with tab_single:
        mode = st.segmented_control(
            "Input mode", ["From dataset", "Manual entry"], default="From dataset", key="fd_mode"
        )
        row_df = pd.DataFrame()
        if mode == "From dataset":
            st.caption("Enter a transaction ID from the selected split.")
            chosen_id = st.text_input("Transaction ID", key="fd_single_id")
            recent_cols = [
                c for c in ["transaction_id", "timestamp", "amount_bdt", "transaction_type", "is_fraud"] if c in df.columns
            ]
            recent = df.sort_values("timestamp", ascending=False).head(10)[recent_cols] if "timestamp" in df.columns else df.head(10)[recent_cols]
            st.dataframe(
                add_status(recent),
                hide_index=True,
                column_config=configured_columns(add_status(recent)),
                alt="Recent transactions available for single scoring",
            )
            if chosen_id:
                row_df = df[df["transaction_id"].astype(str) == chosen_id.strip()].head(1).copy()
                if row_df.empty:
                    st.warning("No transaction with that ID exists in the selected split.")
        else:
            st.caption(f"All {len(feature_cols)} numeric model features are prefilled with split medians.")
            with st.form("fd_manual_form"):
                values = {}
                for offset in range(0, len(feature_cols), 6):
                    chunk = feature_cols[offset : offset + 6]
                    cols = st.columns(len(chunk))
                    for col, name in zip(cols, chunk):
                        default = float(medians.get(name, 0.0))
                        values[name] = col.number_input(
                            name.replace("_", " ").title(),
                            value=default,
                            step=0.01 if abs(default) < 100 else 1.0,
                            format="%.4f",
                        )
                submitted = st.form_submit_button("Score transaction", width="stretch")
            if submitted:
                row_df = pd.DataFrame([values], columns=feature_cols)

        if not row_df.empty:
            scored_row = score_frame(row_df)
            probability = float(scored_row["fraud_probability"].iloc[0])
            rinfo = risk_info(probability)
            r1, r2, r3 = st.columns(3)
            kpi(r1, "Fraud probability", f"{probability:.4f}")
            kpi(r2, "Risk score", str(rinfo["risk_score"]) if rinfo else "—")
            kpi(r3, "Risk level", rinfo["risk_category"] if rinfo else "Unavailable")
            if rinfo:
                st.markdown(f"**Recommended action:** {rinfo['review_action']}")
            else:
                st.info("The project risk-scoring helper could not be loaded.", icon=":material/info:")
            if st.toggle("Show SHAP explanation", key="fd_single_shap"):
                explanation = shap_explanation_for_row(row_df.iloc[0], df)
                if explanation and (explanation["top_positive_features"] or explanation["top_negative_features"]):
                    st.markdown("**Features increasing fraud risk**")
                    st.table(
                        pd.DataFrame(
                            explanation["top_positive_features"], columns=["Feature", "SHAP value"]
                        ),
                        alt="Features increasing the fraud prediction",
                    )
                    st.markdown("**Features decreasing fraud risk**")
                    st.table(
                        pd.DataFrame(
                            explanation["top_negative_features"], columns=["Feature", "SHAP value"]
                        ),
                        alt="Features decreasing the fraud prediction",
                    )
                else:
                    st.info("A SHAP explanation is not available for this prediction.", icon=":material/info:")

    with tab_batch:
        st.caption(
            "Upload a CSV with the model feature columns, or with transaction_id values that exist in this split. "
            f"At most {MAX_BATCH_ROWS:,} rows are scored per run."
        )
        uploaded = st.file_uploader("Batch CSV", type=["csv"], key="fd_batch_upload")
        if uploaded is not None:
            try:
                incoming = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"Could not read the uploaded CSV: {exc}")
                return
            st.caption(f"Uploaded {fmt_int(len(incoming))} rows, {len(incoming.columns)} columns.")
            limit = st.slider("Rows to score", min_value=1, max_value=min(MAX_BATCH_ROWS, len(incoming)), value=min(200, len(incoming)), key="fd_batch_limit")
            incoming = incoming.head(limit)
            missing = [c for c in feature_cols if c not in incoming.columns]
            if not missing:
                batch = incoming.copy()
            elif "transaction_id" in incoming.columns:
                ids = incoming["transaction_id"].astype(str)
                batch = df[df["transaction_id"].astype(str).isin(ids)].head(limit).copy()
                if batch.empty:
                    st.error("None of the uploaded transaction IDs exist in the selected split.")
                    return
                st.caption(f"Matched {fmt_int(len(batch))} transaction IDs against the selected split.")
            else:
                st.error(
                    f"Uploaded CSV is missing {len(missing)} model feature columns and has no usable transaction_id. "
                    f"First missing column: {missing[0]}"
                )
                return
            scored_batch = score_frame(batch)
            scored_batch["risk_score"] = scored_batch["fraud_probability"].apply(
                lambda p: (risk_info(p) or {}).get("risk_score")
            )
            show = [
                c
                for c in [
                    "transaction_id",
                    "timestamp",
                    "amount_bdt",
                    "transaction_type",
                    "is_fraud",
                    "fraud_probability",
                    "risk_score",
                    "risk_level",
                ]
                if c in scored_batch.columns
            ]
            st.dataframe(
                scored_batch[show],
                hide_index=True,
                column_config=configured_columns(scored_batch),
                alt="Batch prediction results with fraud probability and risk level",
            )
            csv_download(scored_batch[show], "Download batch predictions (CSV)", "batch_predictions.csv", "fd_batch_export")


# --------------------------------------------------------------------------- #
# Page: Fraud alerts
# --------------------------------------------------------------------------- #
def render_alerts(df: pd.DataFrame, split_label: str) -> None:
    st.title("Fraud alerts")
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

    scored_full = score_frame(sample)
    keep_cols = display_cols + [c for c in ["fraud_probability", "risk_level"] if c in scored_full.columns]
    scored = scored_full[keep_cols].copy()

    high_critical = int(scored["risk_level"].isin(["High", "Critical"]).sum()) if _artifacts_ready() else 0

    a1, a2, a3, a4 = st.columns(4)
    kpi(a1, "Confirmed fraud alerts", fmt_int(fraud_count))
    kpi(a2, "Fraud rate", f"{fraud_rate:.2%}")
    kpi(a3, "High / critical (scored sample)", fmt_int(high_critical) if _artifacts_ready() else "—")
    kpi(a4, "Average fraud amount", fmt_bdt(avg_fraud))

    st.subheader(":material/warning: Risk tiers")
    tier_counts = scored["risk_level"].value_counts().reindex(RISK_ORDER, fill_value=0)
    tier_cols = st.columns(4)
    for col, level in zip(tier_cols, RISK_ORDER):
        kpi(col, f"{level} risk", fmt_int(tier_counts[level]) if _artifacts_ready() else "—")

    if not _artifacts_ready():
        st.info(
            "Model artifacts are unavailable, so risk levels and fraud probabilities cannot be computed. "
            "Alerts below are the dataset's confirmed fraud transactions.",
            icon=":material/info:",
        )
    st.caption(
        "Alerts are derived from the dataset label and model risk level. "
        "This project has no alert-status workflow or notification backend."
    )

    st.divider()
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader(":material/bar_chart: Risk level distribution")
        if _artifacts_ready():
            counts = tier_counts.reset_index()
            counts.columns = ["risk_level", "alerts"]
            fig = px.bar(
                counts,
                x="risk_level",
                y="alerts",
                labels={"risk_level": "Risk level", "alerts": "Alerts"},
                color="risk_level",
                color_discrete_map=RISK_COLORS,
                category_orders={"risk_level": RISK_ORDER},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig), width="stretch", alt="Fraud alert counts by model risk level")
            st.caption(f"Risk levels computed by the model on {fmt_int(len(scored))} scored fraud transactions.")
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
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(style_fig(fig), width="stretch", alt="Confirmed fraud transactions by fraud type")
        else:
            st.info("Fraud type column is not available.")

    st.subheader(":material/list_alt: Fraud alert details")
    f1, f2, f3 = st.columns(3)
    with f1:
        if _artifacts_ready():
            level_options = [lvl for lvl in RISK_ORDER if lvl in scored["risk_level"].unique()]
            selected_levels = st.multiselect("Risk level", level_options, default=level_options, key="al_levels")
        else:
            selected_levels = []
    with f2:
        if "fraud_type" in scored.columns:
            ftypes = sorted(scored["fraud_type"].dropna().unique().tolist())
            selected_ftypes = st.multiselect("Fraud type", ftypes, default=ftypes, key="al_ftypes")
        else:
            selected_ftypes = []
    with f3:
        selected_status = st.selectbox("Alert status", ["All", "Confirmed fraud"], key="al_status")
        min_prob = (
            st.slider("Minimum fraud probability", 0.0, 1.0, 0.0, 0.05, key="al_min_prob")
            if _artifacts_ready()
            else 0.0
        )

    table = scored
    if selected_levels and "risk_level" in table.columns:
        table = table[table["risk_level"].isin(selected_levels)]
    if selected_ftypes and "fraud_type" in table.columns:
        table = table[table["fraud_type"].isin(selected_ftypes)]
    if selected_status == "Confirmed fraud" and "is_fraud" in table.columns:
        table = table[table["is_fraud"] == 1]
    if _artifacts_ready() and "fraud_probability" in table.columns:
        table = table[table["fraud_probability"].fillna(0.0) >= min_prob]

    if table.empty:
        st.warning("No fraud alerts match the current filters.")
        return

    table = add_status(table)
    st.dataframe(
        table,
        hide_index=True,
        column_config=configured_columns(table),
        alt="Fraud alerts matching the selected filters",
    )

    with st.expander("Alert history"):
        history = apply_date_filter(table, "al_history")
        if "timestamp" in history.columns:
            history = history.sort_values("timestamp", ascending=False)
        st.caption(f"{fmt_int(len(history))} alerts in the selected period.")
        st.dataframe(
            history.head(MAX_SCORE_ROWS),
            hide_index=True,
            column_config=configured_columns(history),
            alt="Chronological fraud alert history",
        )

    for level in RISK_ORDER:
        level_table = table[table["risk_level"] == level] if "risk_level" in table.columns else pd.DataFrame()
        if level_table.empty:
            continue
        with st.expander(f"{level} risk alerts ({fmt_int(len(level_table))})"):
            st.dataframe(
                level_table.head(50),
                hide_index=True,
                column_config=configured_columns(level_table),
                alt=f"{level} risk fraud alerts",
            )

    csv_download(table, "Download alerts (CSV)", "fraud_alerts.csv", "al_export_csv")


# --------------------------------------------------------------------------- #
# Page: Fraud analytics
# --------------------------------------------------------------------------- #
def render_analytics(df: pd.DataFrame, split_label: str) -> None:
    st.title("Fraud analytics")
    st.caption(f"Interactive Plotly analytics on the {split_label} split")

    scoped = df
    selected_types = type_filter(scoped, "an")
    if selected_types and "transaction_type" in scoped.columns:
        scoped = scoped[scoped["transaction_type"].isin(selected_types)]
    scoped = apply_date_filter(scoped, "an")
    st.caption(f"Analysing {fmt_int(len(scoped))} transactions after filters.")
    if scoped.empty:
        st.warning("No transactions match the selected filters.")
        return

    fraud = scoped[scoped["is_fraud"] == 1] if "is_fraud" in scoped.columns else scoped.iloc[0:0]

    st.subheader(":material/donut_large: Fraud distribution")
    col1, col2 = st.columns([1, 2])
    with col1:
        fig = px.pie(
            names=["Legitimate", "Fraud"],
            values=[len(scoped) - len(fraud), len(fraud)],
            hole=0.55,
            color=["Legitimate", "Fraud"],
            color_discrete_map={"Legitimate": COLOR_LEGIT, "Fraud": COLOR_FRAUD},
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig, width="stretch", alt="Share of fraud and legitimate transactions")
    with col2:
        st.subheader(":material/bar_chart: Fraud by transaction type")
        if "transaction_type" in scoped.columns:
            by_type = (
                scoped.groupby("transaction_type")
                .agg(transactions=("is_fraud", "size"), fraud=("is_fraud", "sum"))
                .reset_index()
            )
            by_type["fraud_rate"] = by_type["fraud"] / by_type["transactions"]
            fig = go.Figure()
            fig.add_bar(x=by_type["transaction_type"], y=by_type["fraud"], name="Fraud", marker_color=COLOR_FRAUD)
            fig.add_bar(
                x=by_type["transaction_type"],
                y=by_type["transactions"] - by_type["fraud"],
                name="Legitimate",
                marker_color=COLOR_LEGIT,
            )
            fig.update_layout(barmode="stack")
            st.plotly_chart(style_fig(fig), width="stretch", alt="Fraud and legitimate counts by transaction type")
        else:
            st.info("Transaction type column is not available.")

    st.subheader(":material/query_stats: Fraud by amount range")
    if "amount_bdt" in scoped.columns and not fraud.empty:
        work = scoped[["amount_bdt", "is_fraud"]].copy()
        work["amount_range"] = pd.cut(work["amount_bdt"], bins=AMOUNT_BINS, labels=AMOUNT_LABELS, right=False)
        by_amount = work.groupby("amount_range", observed=True).agg(
            transactions=("is_fraud", "size"), fraud=("is_fraud", "sum")
        ).reset_index()
        by_amount["fraud_rate"] = by_amount["fraud"] / by_amount["transactions"]
        fig = go.Figure()
        fig.add_bar(
            x=by_amount["amount_range"].astype(str),
            y=by_amount["fraud"],
            name="Fraud",
            marker_color=COLOR_FRAUD,
        )
        fig.add_scatter(
            x=by_amount["amount_range"].astype(str),
            y=by_amount["fraud_rate"],
            name="Fraud rate",
            mode="lines+markers",
            line=dict(color=COLOR_PRIMARY, width=3),
            yaxis="y2",
        )
        fig.update_layout(
            yaxis=dict(title="Fraud transactions"),
            yaxis2=dict(title="Fraud rate", overlaying="y", side="right", tickformat=".1%"),
        )
        st.plotly_chart(style_fig(fig), width="stretch", alt="Fraud count and rate by transaction amount range")
    else:
        st.info("Amount column is not available in the selected data.")

    st.subheader(":material/schedule: Hourly fraud analysis")
    if "hour" in scoped.columns:
        hourly = scoped.groupby("hour").agg(transactions=("is_fraud", "size"), fraud=("is_fraud", "sum")).reset_index()
        hourly["fraud_rate"] = hourly["fraud"] / hourly["transactions"]
        fig = go.Figure()
        fig.add_bar(x=hourly["hour"], y=hourly["fraud"], name="Fraud", marker_color=COLOR_FRAUD)
        fig.add_scatter(
            x=hourly["hour"],
            y=hourly["fraud_rate"],
            name="Fraud rate",
            mode="lines+markers",
            line=dict(color=COLOR_PRIMARY, width=3),
            yaxis="y2",
        )
        fig.update_layout(
            xaxis=dict(dtick=1),
            yaxis=dict(title="Fraud transactions"),
            yaxis2=dict(title="Fraud rate", overlaying="y", side="right", tickformat=".1%"),
        )
        st.plotly_chart(style_fig(fig), width="stretch", alt="Fraud volume and rate by hour of day")
    else:
        st.info("Hour column is not available in the selected data.")

    st.subheader(":material/trending_up: Daily and monthly fraud trends")
    if "timestamp" in scoped.columns:
        daily = (
            scoped.groupby(scoped["timestamp"].dt.date)
            .agg(transactions=("is_fraud", "size"), fraud=("is_fraud", "sum"))
            .reset_index()
            .rename(columns={"timestamp": "date"})
        )
        monthly = (
            scoped.groupby(scoped["timestamp"].dt.to_period("M").astype(str))
            .agg(transactions=("is_fraud", "size"), fraud=("is_fraud", "sum"))
            .reset_index()
            .rename(columns={"timestamp": "month"})
        )
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("**Daily**")
            fig = px.line(daily, x="date", y=["fraud", "transactions"], labels={"value": "Transactions", "variable": "Series"})
            st.plotly_chart(style_fig(fig, 320), width="stretch", alt="Daily fraud and transaction trends")
        with tc2:
            st.markdown("**Monthly**")
            fig = px.line(monthly, x="month", y=["fraud", "transactions"], labels={"value": "Transactions", "variable": "Series"})
            st.plotly_chart(style_fig(fig, 320), width="stretch", alt="Monthly fraud and transaction trends")
    else:
        st.info("Timestamp column is not available in the selected data.")

    st.subheader(":material/payments: Fraud amount analysis")
    if "amount_bdt" in fraud.columns:
        ac1, ac2 = st.columns(2)
        with ac1:
            if "transaction_type" in fraud.columns:
                amount_by_type = (
                    fraud.groupby("transaction_type")["amount_bdt"]
                    .agg(["sum", "mean", "count"])
                    .reset_index()
                    .sort_values("sum", ascending=False)
                )
                fig = px.bar(
                    amount_by_type,
                    x="transaction_type",
                    y="sum",
                    labels={"transaction_type": "Transaction type", "sum": "Fraud amount (BDT)"},
                    color_discrete_sequence=[COLOR_FRAUD],
                )
                st.plotly_chart(style_fig(fig, 320), width="stretch", alt="Total fraud amount by transaction type")
            else:
                st.info("Transaction type column is not available.")
        with ac2:
            fig = px.histogram(
                fraud[["amount_bdt"]].head(MAX_SCORE_ROWS),
                x="amount_bdt",
                nbins=40,
                labels={"amount_bdt": "Fraud amount (BDT)"},
                color_discrete_sequence=[COLOR_FRAUD],
            )
            st.plotly_chart(style_fig(fig, 320), width="stretch", alt="Distribution of fraud transaction amounts")
        st.caption(f"Total fraud amount: {fmt_bdt(fraud['amount_bdt'].sum())}")
    else:
        st.info("Amount column is not available in the selected data.")

    st.subheader(":material/psychology: Transaction behavior analysis")
    flags = [c for c in ["new_device", "unusual_location", "sim_changed", "high_value_tx", "is_night", "rapid_tx_flag"] if c in scoped.columns]
    if flags:
        rows = []
        for flag in flags:
            for value, group in scoped.groupby(flag):
                rows.append(
                    {
                        "behavior": f"{flag.replace('_', ' ').title()} = {value}",
                        "transactions": len(group),
                        "fraud": int(group["is_fraud"].sum()),
                        "fraud_rate": float(group["is_fraud"].mean()),
                    }
                )
        behavior = pd.DataFrame(rows)
        fig = px.bar(
            behavior,
            x="behavior",
            y="fraud_rate",
            color="fraud_rate",
            color_continuous_scale=[[0, COLOR_LEGIT], [1, COLOR_FRAUD]],
            labels={"behavior": "Behavior flag", "fraud_rate": "Fraud rate"},
        )
        fig.update_layout(xaxis_tickangle=-30)
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(style_fig(fig), width="stretch", alt="Fraud rate by transaction behavior flag")
    else:
        st.info("No behavior flag columns are available in this dataset.")
    if "failed_login_attempts" in scoped.columns:
        work = scoped[["failed_login_attempts", "is_fraud"]].copy()
        work["attempts"] = pd.cut(
            work["failed_login_attempts"], bins=[-1, 0, 1, 2, 4, np.inf], labels=["0", "1", "2", "3–4", "5+"]
        )
        attempts = work.groupby("attempts", observed=True).agg(
            transactions=("is_fraud", "size"), fraud=("is_fraud", "sum")
        ).reset_index()
        attempts["fraud_rate"] = attempts["fraud"] / attempts["transactions"]
        fig = px.bar(
            attempts,
            x="attempts",
            y="fraud_rate",
            labels={"attempts": "Failed login attempts", "fraud_rate": "Fraud rate"},
            color_discrete_sequence=[COLOR_FRAUD],
        )
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(style_fig(fig, 320), width="stretch", alt="Fraud rate by failed login attempts")


# --------------------------------------------------------------------------- #
# Page: Location analytics
# --------------------------------------------------------------------------- #
def render_location_analytics(df: pd.DataFrame, split_label: str) -> None:
    st.title("Location analytics")
    st.caption("Geographic fields available in the processed dataset: sender/receiver division and district")

    if "sender_division" not in df.columns and "receiver_division" not in df.columns:
        st.error("No location columns are available in the selected dataset split.")
        return

    scoped = apply_date_filter(df, "loc")
    if "sender_division" in scoped.columns:
        options = sorted(scoped["sender_division"].dropna().unique().tolist())
        picked = st.multiselect("Sender division", options, default=options, key="loc_divisions")
        if picked:
            scoped = scoped[scoped["sender_division"].isin(picked)]
    if scoped.empty:
        st.warning("No transactions match the selected filters.")
        return

    st.subheader(":material/public: Division-wise fraud analysis")
    if "sender_division" in scoped.columns:
        by_division = (
            scoped.groupby("sender_division")
            .agg(transactions=("is_fraud", "size"), fraud=("is_fraud", "sum"))
            .reset_index()
            .sort_values("fraud", ascending=False)
        )
        by_division["fraud_rate"] = by_division["fraud"] / by_division["transactions"]
        fig = go.Figure()
        fig.add_bar(
            x=by_division["sender_division"],
            y=by_division["fraud"],
            name="Fraud",
            marker_color=COLOR_FRAUD,
        )
        fig.add_scatter(
            x=by_division["sender_division"],
            y=by_division["fraud_rate"],
            name="Fraud rate",
            mode="lines+markers",
            line=dict(color=COLOR_PRIMARY, width=3),
            yaxis="y2",
        )
        fig.update_layout(
            yaxis=dict(title="Fraud transactions"),
            yaxis2=dict(title="Fraud rate", overlaying="y", side="right", tickformat=".1%"),
        )
        st.plotly_chart(style_fig(fig), width="stretch", alt="Fraud count and rate by sender division")
    else:
        st.info("Sender division is not available.")

    st.subheader(":material/location_on: District-wise fraud analysis")
    if "sender_district" in scoped.columns:
        top_n = st.slider("Top districts", min_value=5, max_value=50, value=15, step=5, key="loc_top_n")
        by_district = (
            scoped.groupby("sender_district")
            .agg(transactions=("is_fraud", "size"), fraud=("is_fraud", "sum"))
            .reset_index()
            .sort_values("fraud", ascending=False)
            .head(top_n)
        )
        by_district["fraud_rate"] = by_district["fraud"] / by_district["transactions"]
        fig = px.bar(
            by_district.iloc[::-1],
            x="fraud",
            y="sender_district",
            orientation="h",
            color="fraud_rate",
            color_continuous_scale=[[0, COLOR_LEGIT], [1, COLOR_FRAUD]],
            labels={"fraud": "Fraud transactions", "sender_district": "Sender district", "fraud_rate": "Fraud rate"},
        )
        st.plotly_chart(style_fig(fig, max(380, top_n * 22)), width="stretch", alt="Top sender districts by fraud count")
    else:
        st.info("Sender district is not available.")

    st.subheader(":material/bar_chart: Location-wise transaction distribution")
    if {"sender_division", "sender_district"}.issubset(scoped.columns):
        locations = (
            scoped.groupby(["sender_division", "sender_district"])
            .agg(transactions=("is_fraud", "size"), fraud=("is_fraud", "sum"))
            .reset_index()
        )
        locations["fraud_rate"] = locations["fraud"] / locations["transactions"]
        fig = px.treemap(
            locations,
            path=["sender_division", "sender_district"],
            values="transactions",
            color="fraud_rate",
            color_continuous_scale=[[0, COLOR_LEGIT], [1, COLOR_FRAUD]],
            hover_data=["fraud"],
        )
        st.plotly_chart(style_fig(fig, 460), width="stretch", alt="Transaction volume and fraud rate by sender location")
        st.caption(
            "The dataset has no coordinates or map boundaries, so the geographic view is an interactive "
            "location hierarchy instead of a coordinate map."
        )
    else:
        st.info("Division and district columns are required for the location hierarchy.")

    st.subheader(":material/compare_arrows: Sender vs receiver location comparison")
    sender_col = "sender_division" if "sender_division" in scoped.columns else None
    receiver_col = "receiver_division" if "receiver_division" in scoped.columns else None
    if sender_col and receiver_col:
        sender = scoped.groupby(sender_col)["is_fraud"].agg(["sum", "size"]).reset_index()
        sender.columns = ["division", "fraud", "transactions"]
        sender["side"] = "Sender"
        receiver = scoped.groupby(receiver_col)["is_fraud"].agg(["sum", "size"]).reset_index()
        receiver.columns = ["division", "fraud", "transactions"]
        receiver["side"] = "Receiver"
        both = pd.concat([sender, receiver], ignore_index=True)
        fig = px.bar(
            both,
            x="division",
            y="fraud",
            color="side",
            barmode="group",
            labels={"division": "Division", "fraud": "Fraud transactions", "side": "Location side"},
        )
        st.plotly_chart(style_fig(fig), width="stretch", alt="Fraud counts by sender and receiver division")
    else:
        st.info("Sender and receiver division columns are required for this comparison.")

    st.subheader(":material/leaderboard: Top fraud locations")
    group_cols = [c for c in ["sender_division", "sender_district"] if c in scoped.columns]
    if group_cols:
        top_locations = (
            scoped.groupby(group_cols)
            .agg(
                transactions=("is_fraud", "size"),
                fraud=("is_fraud", "sum"),
                fraud_amount=("amount_bdt", "sum") if "amount_bdt" in scoped.columns else ("is_fraud", "sum"),
            )
            .reset_index()
            .sort_values("fraud", ascending=False)
            .head(20)
        )
        top_locations["fraud_rate"] = top_locations["fraud"] / top_locations["transactions"]
        st.dataframe(
            top_locations,
            hide_index=True,
            column_config=configured_columns(top_locations),
            alt="Top locations by confirmed fraud transactions",
        )
        csv_download(top_locations, "Download top fraud locations (CSV)", "top_fraud_locations.csv", "loc_export_csv")
    else:
        st.info("No location columns are available for ranking.")


# --------------------------------------------------------------------------- #
# Page: Model performance (existing functionality)
# --------------------------------------------------------------------------- #
def render_model_performance(show_title: bool = True) -> None:
    if show_title:
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
    model_name = st.selectbox("Model", model_names, index=default_idx, key="mp_model")

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
            st.caption(f"TP={fmt_int(tp)} • FP={fmt_int(fp)} • FN={fmt_int(fn)} • TN={fmt_int(tn)}")
        else:
            st.info("Confusion matrix values are not available for this model (unsupervised).")

    with col_right:
        st.subheader(":material/leaderboard: Model comparison")
        if not comparison.empty and {"model", "f1"}.issubset(comparison.columns):
            metric_cols = [c for c in ["precision", "recall", "f1", "roc_auc", "pr_auc"] if c in comparison.columns]
            melted = comparison.melt(id_vars="model", value_vars=metric_cols, var_name="metric", value_name="score")
            melted = melted[melted["score"] > 0]
            fig = px.bar(
                melted,
                x="model",
                y="score",
                color="metric",
                barmode="group",
                labels={"model": "Model", "score": "Score", "metric": "Metric"},
            )
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(style_fig(fig), width="stretch", alt="Comparison of model metrics across all trained models")
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
        st.plotly_chart(style_fig(fig), width="stretch", alt="Top feature importances for the selected model")


# --------------------------------------------------------------------------- #
# Page: Model intelligence
# --------------------------------------------------------------------------- #
def render_model_intelligence() -> None:
    st.title("Model intelligence")
    arts = load_artifacts()
    results = load_results_json()
    st.caption("Existing trained model and evaluation artifacts — nothing is retrained on this page.")

    if not arts.get("model_name"):
        st.error("No active model artifact was found.")
    else:
        with st.container(border=True):
            st.subheader(":material/model_training: Active model")
            st.markdown(f"**{arts['model_name']}**")
            st.caption(f"Scaler: {'loaded' if arts.get('scaler') else 'unavailable'} • Feature count: {len(_model_feature_columns(arts))}")
            if arts.get("error"):
                st.caption(f"Artifact details: {arts['error']}")
            if results:
                best_name = max(results, key=lambda name: float((results[name].get("test_metrics") or {}).get("f1", 0) or 0))
                best_f1 = float((results[best_name].get("test_metrics") or {}).get("f1", 0) or 0)
                st.caption(f"Best stored test F1: {best_name} ({best_f1:.4f})")

    st.divider()
    render_model_performance(show_title=False)

    st.divider()
    st.subheader(":material/science: SHAP explanation")
    if st.toggle("Compute SHAP global importance on a test sample", key="mi_shap_toggle"):
        with st.spinner("Computing SHAP values with the project's existing explainer..."):
            importance = shap_global_importance(load_split("test").head(100))
        if importance.empty:
            st.info("SHAP global importance could not be computed with the current model artifacts.")
        else:
            top = importance.head(15).iloc[::-1]
            fig = px.bar(
                top,
                x="mean_shap_abs",
                y="feature",
                orientation="h",
                labels={"mean_shap_abs": "Mean |SHAP|", "feature": "Feature"},
                color_discrete_sequence=[COLOR_PRIMARY],
            )
            st.plotly_chart(style_fig(fig), width="stretch", alt="Global SHAP importance on a test sample")
            st.caption("Computed on the first 100 test rows using `src.explainability.shap_explainer`.")
    else:
        st.caption("Enable the toggle to compute SHAP values on demand using the existing explainer module.")


# --------------------------------------------------------------------------- #
# Page: Reports
# --------------------------------------------------------------------------- #
def render_reports(df: pd.DataFrame, split_label: str) -> None:
    st.title("Reports")
    st.caption(f"Aggregated reports from the {split_label} split")

    tabs = st.tabs(["Fraud summary", "Daily & monthly", "High-risk", "Filtered export"])
    split_key = SPLITS[split_label]

    with tabs[0]:
        total = len(df)
        fraud_count = int(df["is_fraud"].sum()) if "is_fraud" in df.columns else 0
        fraud_amount = (
            float(df.loc[df["is_fraud"] == 1, "amount_bdt"].sum())
            if fraud_count and "amount_bdt" in df.columns
            else 0.0
        )
        summary = pd.DataFrame(
            {
                "Metric": [
                    "Total transactions",
                    "Fraud transactions",
                    "Legitimate transactions",
                    "Fraud rate",
                    "Total fraud amount (BDT)",
                    "Average fraud amount (BDT)",
                ],
                "Value": [
                    str(total),
                    str(fraud_count),
                    str(total - fraud_count),
                    f"{fraud_count / total:.2%}" if total else "—",
                    f"{fraud_amount:,.2f}",
                    f"{fraud_amount / fraud_count:,.2f}" if fraud_count else "—",
                ],
            }
        )
        st.dataframe(summary, hide_index=True, alt="Fraud summary metrics")
        csv_download(summary, "Download fraud summary (CSV)", f"fraud_summary_{split_key}.csv", "rp_fraud_summary_csv")
        if "transaction_type" in df.columns:
            by_type = (
                df.groupby("transaction_type")
                .agg(transactions=("is_fraud", "size"), fraud=("is_fraud", "sum"))
                .reset_index()
            )
            by_type["fraud_rate"] = by_type["fraud"] / by_type["transactions"]
            st.subheader("Transaction summary by type")
            st.dataframe(by_type, hide_index=True, alt="Transaction and fraud counts by type")
            csv_download(by_type, "Download transaction summary (CSV)", f"transaction_summary_{split_key}.csv", "rp_tx_summary_csv")

    with tabs[1]:
        if "timestamp" in df.columns:
            daily = (
                df.groupby(df["timestamp"].dt.date)
                .agg(transactions=("is_fraud", "size"), fraud=("is_fraud", "sum"))
                .reset_index()
                .rename(columns={"timestamp": "date"})
            )
            monthly = (
                df.groupby(df["timestamp"].dt.to_period("M").astype(str))
                .agg(transactions=("is_fraud", "size"), fraud=("is_fraud", "sum"))
                .reset_index()
                .rename(columns={"timestamp": "month"})
            )
            rc1, rc2 = st.columns(2)
            with rc1:
                st.subheader("Daily report")
                st.dataframe(daily, hide_index=True, alt="Daily transaction and fraud report")
                csv_download(daily, "Download daily report (CSV)", f"daily_report_{split_key}.csv", "rp_daily_csv")
            with rc2:
                st.subheader("Monthly report")
                st.dataframe(monthly, hide_index=True, alt="Monthly transaction and fraud report")
                csv_download(monthly, "Download monthly report (CSV)", f"monthly_report_{split_key}.csv", "rp_monthly_csv")
        else:
            st.info("Timestamp column is not available in this split.")

    with tabs[2]:
        st.caption("High-risk report is model-scored on the most recent transactions in the split.")
        sample = (
            df.sort_values("timestamp", ascending=False).head(MAX_SCORE_ROWS).copy()
            if "timestamp" in df.columns
            else df.head(MAX_SCORE_ROWS).copy()
        )
        scored = score_frame(sample)
        high_risk = scored[scored["risk_level"].isin(["High", "Critical"])].copy()
        if high_risk.empty:
            st.info("No high or critical risk transactions were found in the scored sample.")
        else:
            show = [
                c
                for c in [
                    "transaction_id",
                    "timestamp",
                    "amount_bdt",
                    "transaction_type",
                    "provider",
                    "fraud_probability",
                    "risk_level",
                    "is_fraud",
                ]
                if c in high_risk.columns
            ]
            st.dataframe(
                high_risk[show],
                hide_index=True,
                column_config=configured_columns(high_risk),
                alt="High and critical risk transactions",
            )
            csv_download(high_risk[show], "Download high-risk report (CSV)", f"high_risk_report_{split_key}.csv", "rp_high_risk_csv")

    with tabs[3]:
        filtered = apply_date_filter(df, "rp")
        selected_types = type_filter(filtered, "rp")
        if selected_types and "transaction_type" in filtered.columns:
            filtered = filtered[filtered["transaction_type"].isin(selected_types)]
        status = st.selectbox("Fraud status", ["All", "Fraud", "Legitimate"], key="rp_status")
        if status != "All" and "is_fraud" in filtered.columns:
            filtered = filtered[filtered["is_fraud"] == (1 if status == "Fraud" else 0)]
        st.caption(f"{fmt_int(len(filtered))} transactions match the export filters.")
        st.dataframe(
            add_status(filtered.head(1000)),
            hide_index=True,
            column_config=configured_columns(add_status(filtered)),
            alt="Preview of the filtered transaction export",
        )
        csv_download(add_status(filtered), "Download filtered transactions (CSV)", f"filtered_transactions_{split_key}.csv", "rp_filtered_csv")

    st.divider()
    st.subheader(":material/picture_as_pdf: PDF export")
    pdf_libraries = [name for name in ["reportlab", "fpdf", "weasyprint"] if importlib.util.find_spec(name)]
    if pdf_libraries:
        st.info(f"A PDF library is available ({', '.join(pdf_libraries)}), but no PDF export has been wired into this dashboard.")
    else:
        st.info("PDF export is unavailable: no PDF library is installed in this project.")


# --------------------------------------------------------------------------- #
# Page entry points
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    df, label = current_split()
    render_overview(df, label)


def page_transactions() -> None:
    df, label = current_split()
    render_transactions(df, label)


def page_fraud_alerts() -> None:
    df, label = current_split()
    render_alerts(df, label)


def page_fraud_detection() -> None:
    df, label = current_split()
    render_fraud_detection(df, label)


def page_fraud_analytics() -> None:
    df, label = current_split()
    render_analytics(df, label)


def page_location_analytics() -> None:
    df, label = current_split()
    render_location_analytics(df, label)


def page_reports() -> None:
    df, label = current_split()
    render_reports(df, label)


# --------------------------------------------------------------------------- #
# App shell
# --------------------------------------------------------------------------- #
def main() -> None:
    if "split_label" not in st.session_state:
        st.session_state["split_label"] = "Train"

    page = st.navigation(
        {
            "": [
                st.Page(
                    page_overview,
                    title="Overview",
                    icon=":material/dashboard:",
                    url_path="overview",
                    help="KPIs, fraud trends and recent suspicious transactions for the selected split.",
                )
            ],
            "Transactions": [
                st.Page(
                    page_transactions,
                    title="Transactions",
                    icon=":material/receipt_long:",
                    url_path="transactions",
                    help="Search, filter, sort, inspect and export transactions from the selected split.",
                ),
                st.Page(
                    page_fraud_alerts,
                    title="Fraud alerts",
                    icon=":material/warning:",
                    url_path="fraud-alerts",
                    help="Confirmed fraud alerts split into model risk tiers with history and CSV export.",
                ),
            ],
            "Detection and analytics": [
                st.Page(
                    page_fraud_detection,
                    title="Fraud detection",
                    icon=":material/radar:",
                    url_path="fraud-detection",
                    help="Score one transaction or a batch CSV with the existing model and SHAP helpers.",
                ),
                st.Page(
                    page_fraud_analytics,
                    title="Fraud analytics",
                    icon=":material/query_stats:",
                    url_path="fraud-analytics",
                    help="Interactive Plotly charts for fraud distribution, trends, amounts and behavior.",
                ),
                st.Page(
                    page_location_analytics,
                    title="Location analytics",
                    icon=":material/public:",
                    url_path="location-analytics",
                    help="Fraud and transaction patterns by sender and receiver division and district.",
                ),
            ],
            "Insights": [
                st.Page(
                    render_model_intelligence,
                    title="Model intelligence",
                    icon=":material/model_training:",
                    url_path="model-intelligence",
                    help="Existing model metrics, comparison, feature importance and on-demand SHAP values.",
                ),
                st.Page(
                    page_reports,
                    title="Reports",
                    icon=":material/description:",
                    url_path="reports",
                    help="Fraud, transaction, trend and high-risk reports with CSV downloads.",
                ),
            ],
        },
        position="sidebar",
        expanded=True,
    )

    with st.sidebar:
        st.divider()
        st.caption(":material/shield: AI-MFS Fraud Detection")
        st.selectbox("Dataset split", list(SPLITS.keys()), key="split_label")
        st.divider()
        arts = load_artifacts()
        if _artifacts_ready():
            st.success(f"Model: {arts.get('model_name')}", icon=":material/check_circle:")
        else:
            st.warning("Model artifacts unavailable", icon=":material/warning:")
            if arts.get("error"):
                st.caption(f"Details: {arts['error']}")
        st.caption("Synthetic academic dataset • Bangladesh MFS ecosystem")

    try:
        page.run()
    except Exception as exc:
        st.error("The dashboard encountered an unexpected error.")
        st.exception(exc)


if __name__ == "__main__":
    main()
