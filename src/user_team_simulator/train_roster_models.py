# src/user_team_simulator/train_roster_models.py
import argparse
import glob
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from evaluate_and_log_rosters import log_model_result

# ===== Config =====
DEFAULT_DATA_GLOB = "data/processed/roster_training_set.csv"   # or glob like data/processed/team_roster_features_*.csv
TARGET = "won_championship"
DROP_COLS = ["TEAM_NAME", "season"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
BEST_MODEL_OUT = "models/roster_best_model.pkl"
# ==================


def ensure_dirs():
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)


def load_frames(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if TARGET not in df.columns:
            raise ValueError(f"'{TARGET}' missing in {p}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    print(f"📦 Loaded {len(paths)} file(s), {len(out)} rows total.")
    return out


def select_features(df: pd.DataFrame):
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns] + [TARGET], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy().fillna(0)
    y = df[TARGET].astype(int)
    return X, y, list(X.columns)


def get_probas(model, X):
    # works with Pipelines too
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        # Fallback: map decision_function to [0,1] via sigmoid
        from scipy.special import expit
        return expit(model.decision_function(X))
    # last resort: hard preds as pseudo-proba
    return model.predict(X).astype(float)


def eval_and_log(name, model, X_test, y_test, params, threshold, results_path=None):
    y_proba = get_probas(model, X_test)
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="binary", zero_division=0)

    prec_pos = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_pos  = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    cm       = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n🔧 {name}")
    print(f"Threshold: {threshold:.2f}  |  Accuracy: {acc:.4f}  |  F1(+): {f1:.4f}  |  Prec(+): {prec_pos:.4f}  |  Rec(+): {rec_pos:.4f}")
    print("Confusion matrix:\n", np.array(cm))
    print("Classification report:\n", classification_report(y_test, y_pred, zero_division=0))
    class_counts = pd.Series(y_test).value_counts().astype(int).to_dict()

    extras = {
        "precision_pos": round(float(prec_pos), 4),
        "recall_pos": round(float(rec_pos), 4),
        "threshold": float(threshold),
        "confusion_matrix": cm,  # already Python lists/ints
        "class_counts_test": class_counts
    }
    
    
    # extras = {
    #     "precision_pos": round(float(prec_pos), 4),
    #     "recall_pos": round(float(rec_pos), 4),
    #     "threshold": threshold,
    #     "confusion_matrix": cm,
    #     "class_counts_test": dict(pd.Series(y_test).value_counts())
    # }
    
    
    log_model_result(name, acc, f1, params=params, extra=extras, results_path=results_path)
    return acc, f1


def maybe_smote(X_train, y_train, use_smote: bool):
    if not use_smote:
        return X_train, y_train
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res


def main():
    parser = argparse.ArgumentParser(description="Train roster-level models with imbalance fixes.")
    parser.add_argument("--data", default=DEFAULT_DATA_GLOB, help="CSV or glob")
    parser.add_argument("--smote", action="store_true", help="Apply SMOTE on train only")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for positive class")
    parser.add_argument("--results_json", default="results/model_results_part2.json")
    args = parser.parse_args()

    ensure_dirs()
    df = load_frames(args.data)

    print("📊 Class distribution:\n", df[TARGET].value_counts())

    X, y, feature_names = select_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("📐 Feature columns used:", len(feature_names))

    # ---- Logistic Regression (balanced) ----
    log_reg = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="lbfgs", n_jobs=None
        ))
    ])
    lr_params = {"class_weight": "balanced", "solver": "lbfgs", "max_iter": 2000, "smote": args.smote}
    X_res, y_res = maybe_smote(X_train, y_train, args.smote)
    log_reg.fit(X_res, y_res)
    acc_lr, f1_lr = eval_and_log("logistic_roster", log_reg, X_test, y_test, lr_params, args.threshold, args.results_json)

    # ---- Random Forest (balanced) ----
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_params = {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2, "class_weight": "balanced", "smote": args.smote}
    X_res, y_res = maybe_smote(X_train, y_train, args.smote)
    rf.fit(X_res, y_res)
    acc_rf, f1_rf = eval_and_log("random_forest_roster", rf, X_test, y_test, rf_params, args.threshold, args.results_json)

    # ---- XGBoost ----
    results = [("logistic_roster", acc_lr, f1_lr, log_reg),
               ("random_forest_roster", acc_rf, f1_rf, rf)]

    if HAS_XGB:
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        spw = 1.0 if args.smote else (neg / max(pos, 1))

        xgb = XGBClassifier(
            n_estimators=700, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
            min_child_weight=1, eval_metric="logloss",
            tree_method="hist", random_state=RANDOM_STATE,
            scale_pos_weight=spw
        )
        xgb_params = {
            "n_estimators": 700, "max_depth": 3, "learning_rate": 0.05,
            "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 1.0,
            "min_child_weight": 1, "scale_pos_weight": round(spw, 2), "smote": args.smote
        }
        X_res, y_res = maybe_smote(X_train, y_train, args.smote)
        xgb.fit(X_res, y_res)
        acc_xgb, f1_xgb = eval_and_log("xgboost_roster", xgb, X_test, y_test, xgb_params, args.threshold, args.results_json)
        results.append(("xgboost_roster", acc_xgb, f1_xgb, xgb))
    else:
        print("⚠️ XGBoost not installed; skipping. (pip install xgboost)")

    # Pick best by F1 then accuracy
    results.sort(key=lambda r: (r[2], r[1]), reverse=True)
    best_name, best_acc, best_f1, best_model = results[0]
    print(f"\n🏁 Best: {best_name} | acc={best_acc:.4f}  f1={best_f1:.4f}")

    joblib.dump({"model": best_model, "feature_names": feature_names, "tag": best_name}, BEST_MODEL_OUT)
    print(f"✅ Best model saved to {BEST_MODEL_OUT}")


if __name__ == "__main__":
    main()