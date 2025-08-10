# --- path setup must come first ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]   # project root
sys.path.append(str(ROOT))                   # make "src" importable

# std/third‑party
import os
import tempfile
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# project imports (now that sys.path is set)
from src.user_team_simulator.ensembles import SoftVoteEnsemble as _SV
from src.user_team_simulator.roster_features import aggregate_team

# ensure unpickler can resolve SoftVoteEnsemble saved from __main__
setattr(sys.modules[__name__], "SoftVoteEnsemble", _SV)

# ---- Streamlit page config ----
st.set_page_config(page_title="NBA Team Builder — Part 2", page_icon="🏗️", layout="wide")

# ---- Paths / constants ----
DATA_PATH = "data/processed/players_stats_with_salaries_2025_26.csv"
LOCAL_ENSEMBLE = "models/roster_ensemble.pkl"   # saved by calibrate_and_ensemble.py
LOCAL_SINGLE   = "models/roster_best_model.pkl" # fallback
DEFAULT_SALARY_CAP = 140_000_000  # $140M as an example cap

# ---- S3 config (env overrides are handy for EC2 later) ----
S3_BUCKET = os.getenv("S3_BUCKET", "nba-roster-models-allenlu")
S3_KEY_ENSEMBLE = os.getenv("S3_KEY_ENSEMBLE", "models/roster_ensemble.pkl")
S3_KEY_SINGLE   = os.getenv("S3_KEY_SINGLE",   "models/roster_best_model.pkl")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

PRIMARY = "#0EA5E9"  # blue-ish
st.markdown(f"""
<style>
.big-title {{ font-size: 2rem; font-weight: 700; margin-bottom: .25rem; }}
.subtle {{ color: #64748b; }}
.metric-card {{
  padding: 12px 14px; border-radius: 14px; border: 1px solid #e2e8f0;
  background: linear-gradient(135deg, rgba(14,165,233,0.06), rgba(14,165,233,0.02));
}}
.stSelectbox label, .stMultiSelect label {{ font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🏗️ User‑Built Team Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Pick players under a cap, then predict your team’s championship odds.</div>', unsafe_allow_html=True)
st.write("")

@st.cache_data
def load_players():
    df = pd.read_csv(DATA_PATH)
    if "salary_2025_26" in df.columns:
        df["salary_2025_26"] = pd.to_numeric(df["salary_2025_26"], errors="coerce")
    else:
        df["salary_2025_26"] = np.nan

    df["Player_label"] = df["Player"]
    if "Tm" in df.columns:
        df["Player_label"] = df["Player_label"] + " — " + df["Tm"].astype(str)

    return df.sort_values(by=["salary_2025_26", "Player"], ascending=[False, True])

@st.cache_resource(show_spinner="Loading model…")
def load_inference_model():
    """Try S3 (ensemble → single), then local (ensemble → single)."""
    def _download_s3(key, label):
        try:
            import boto3
            s3 = boto3.client("s3", region_name=AWS_REGION)
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                s3.download_fileobj(S3_BUCKET, key, tmp)
                tmp_path = tmp.name
            payload = joblib.load(tmp_path)
            tag = f"s3:{label}"
            return payload, tag
        except Exception as e:
            st.info(f"S3 {label} not available: {e}")
            return None, None

    # 1) S3 ensemble → 2) S3 single
    payload, tag = _download_s3(S3_KEY_ENSEMBLE, "ensemble")
    if payload is None:
        payload, tag = _download_s3(S3_KEY_SINGLE, "single")

    # 3) Local ensemble → 4) Local single
    if payload is None and os.path.exists(LOCAL_ENSEMBLE):
        payload = joblib.load(LOCAL_ENSEMBLE); tag = "local:ensemble"
    if payload is None and os.path.exists(LOCAL_SINGLE):
        payload = joblib.load(LOCAL_SINGLE); tag = "local:single"

    if payload is None:
        raise FileNotFoundError("No model found from S3 or local. Train/upload first.")

    # Support both ensemble and single‑model payload structures
    model = payload.get("ensemble") or payload.get("model") or payload
    feature_names = payload.get("feature_names")
    model_tag = payload.get("tag", tag)

    return model, feature_names, model_tag

players_df = load_players()
model, feature_names, model_tag = load_inference_model()
st.caption(f"📦 Model source: **{model_tag}**")

# --- Sidebar: constraints ---
st.sidebar.header("⚙️ Constraints")
salary_cap = st.sidebar.number_input("Salary Cap (USD)", min_value=50_000_000, max_value=300_000_000, value=DEFAULT_SALARY_CAP, step=1_000_000)
min_players = st.sidebar.slider("Min roster size", 5, 15, 8)
max_players = st.sidebar.slider("Max roster size", 8, 18, 12)

# --- Player selector ---
st.subheader("1) Select Players")
labels = players_df["Player_label"].tolist()
chosen = st.multiselect("Choose your roster:", options=labels, default=[], max_selections=max_players)
chosen_rows = players_df[players_df["Player_label"].isin(chosen)].copy()

# Salary + counts
total_salary = float(chosen_rows["salary_2025_26"].fillna(0).sum())
num_players = len(chosen_rows)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown('<div class="metric-card"><b>Players</b><br>' + f'{num_players} / {max_players}</div>', unsafe_allow_html=True)
with col_b:
    st.markdown('<div class="metric-card"><b>Salary Used</b><br>' + f'${total_salary:,.0f}</div>', unsafe_allow_html=True)
with col_c:
    remaining = salary_cap - total_salary
    color = "red" if remaining < 0 else "black"
    st.markdown(f'<div class="metric-card"><b>Remaining Cap</b><br><span style="color:{color}">${remaining:,.0f}</span></div>', unsafe_allow_html=True)

# --- Predict ---
st.subheader("2) Predict Championship Probability")
disabled = False
msg = None
if num_players < min_players:
    disabled, msg = True, f"Select at least {min_players} players."
elif num_players > max_players:
    disabled, msg = True, f"Reduce to at most {max_players} players."
elif total_salary > salary_cap:
    disabled, msg = True, "Salary exceeds cap."

if msg:
    st.info(msg)

if st.button("🏆 Predict", disabled=disabled):
    feats = aggregate_team(chosen_rows)
    X_team = pd.DataFrame([feats])

    # Align with model feature order
    if feature_names is not None:
        for c in feature_names:
            if c not in X_team.columns:
                X_team[c] = 0.0
        X_team = X_team[feature_names]
    else:
        X_team = X_team.select_dtypes(include=[np.number]).fillna(0.0)

    proba = float(model.predict_proba(X_team)[:, 1][0])
    st.success(f"**Predicted championship probability:** {proba*100:.2f}%  \n_Model: {model_tag}_")

    with st.expander("Show engineered team features"):
        st.dataframe(X_team.T.rename(columns={0: "value"}))

# Optional: a quick “reload model” to clear Streamlit’s cache
if st.sidebar.button("🔄 Reload model"):
    load_inference_model.clear()
    st.experimental_rerun()

st.caption("Tip: search players and tweak the roster to see how the probability moves.")