# src/user_team_simulator/calibrate_and_ensemble.py
import argparse, json, os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# SMOTE (optional)
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False

# our logger
from evaluate_and_log_rosters import log_model_result

RANDOM_STATE = 42
RESULTS_JSON = "results/result_part2.json"
BEST_ENSEMBLE_OUT = "models/roster_ensemble.pkl"
CALIB_DIR = "models/calibrated"


def ensure_dirs():
    Path("models").mkdir(exist_ok=True, parents=True)
    Path(CALIB_DIR).mkdir(exist_ok=True, parents=True)
    Path("results").mkdir(exist_ok=True, parents=True)


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "won_championship" not in df.columns:
        raise ValueError("Column 'won_championship' not found in the dataset.")
    return df


def select_features(df: pd.DataFrame):
    drop_cols = ["TEAM_NAME", "season", "won_championship"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).fillna(0).copy()
    y = df["won_championship"].astype(int).copy()
    return X, y, list(X.columns)


def maybe_smote(X_train, y_train, use_smote: bool):
    if not use_smote:
        return X_train, y_train
    if not HAS_SMOTE:
        print("⚠️ SMOTE requested but not installed; continuing without SMOTE.")
        return X_train, y_train
    sm = SMOTE(random_state=RANDOM_STATE)
    return sm.fit_resample(X_train, y_train)


def build_base_models():
    models = {}

    # Logistic Regression (with scaling in a pipeline)
    log_reg = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])
    models["logistic_roster"] = (log_reg, {"class_weight": "balanced", "solver": "lbfgs", "max_iter": 2000})

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    models["random_forest_roster"] = (rf, {"n_estimators": 700, "max_depth": None, "min_samples_leaf": 2, "class_weight": "balanced"})

    # XGBoost (if available)
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=700,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=1,
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE
        )
        models["xgboost_roster"] = (xgb, {"n_estimators": 700, "max_depth": 3, "learning_rate": 0.05,
                                          "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 1.0,
                                          "min_child_weight": 1})
    else:
        print("⚠️ XGBoost not installed; skipping xgboost_roster.")

    return models


def best_threshold_for_f1(y_true, probs, min_pos=1):
    # guard: if almost all probs identical, fallback to 0.5
    if len(np.unique(probs)) < 3:
        return 0.5, f1_score(y_true, (probs >= 0.5).astype(int), zero_division=0)

    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1s = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    # thresholds array is len-1 compared to precisions/recalls
    if len(thresholds) == 0:
        return 0.5, f1_score(y_true, (probs >= 0.5).astype(int), zero_division=0)

    # pick best f1 threshold but ensure at least one positive predicted
    order = np.argsort(-f1s[:-1])
    for idx in order:
        t = thresholds[idx]
        preds = (probs >= t).astype(int)
        if preds.sum() >= min_pos:
            return float(t), float(f1s[idx])

    # fallback
    return 0.5, f1_score(y_true, (probs >= 0.5).astype(int), zero_division=0)


# def calibrate_model(name, fitted_model, X_val, y_val):
#     """Try isotonic first; if it fails (too few positives), fall back to sigmoid."""
#     # Need some positives to fit isotonic reliably; with tiny y=1, it can fail.
#     try:
#         calib = CalibratedClassifierCV(fitted_model, method="isotonic", cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE))
#         calib.fit(X_val, y_val)
#         method = "isotonic"
#     except Exception:
#         calib = CalibratedClassifierCV(fitted_model, method="sigmoid", cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE))
#         calib.fit(X_val, y_val)
#         method = "sigmoid"

#     # save calibrated model
#     import joblib
#     out_path = os.path.join(CALIB_DIR, f"{name}_calibrated_{method}.pkl")
#     joblib.dump(calib, out_path)
#     return calib, method, out_path

def calibrate_model_prefit(name, fitted_model, X_cal, y_cal):
    """
    Calibrate a prefit model using a separate calibration split.
    Try isotonic; if not enough positives / fails, fall back to sigmoid.
    If both fail, return None to use raw probabilities.
    """
    from sklearn.calibration import CalibratedClassifierCV
    import joblib

    def _try(method):
        calib = CalibratedClassifierCV(fitted_model, method=method, cv="prefit")
        calib.fit(X_cal, y_cal)
        return calib

    # Try isotonic first
    try:
        calib = _try("isotonic")
        method = "isotonic"
    except Exception:
        # Fallback to sigmoid
        try:
            calib = _try("sigmoid")
            method = "sigmoid"
        except Exception:
            return None, "none", None

    out_path = os.path.join(CALIB_DIR, f"{name}_calibrated_{method}.pkl")
    joblib.dump(calib, out_path)
    return calib, method, out_path

class SoftVoteEnsemble:
    """Tiny soft-vote ensemble over calibrated models."""
    def __init__(self, models, weights=None):
        self.models = models  # list of fitted models with predict_proba
        if weights is None:
            self.weights = np.ones(len(models)) / len(models)
        else:
            w = np.array(weights, dtype=float)
            self.weights = w / w.sum()

    def predict_proba(self, X):
        probs = []
        for m in self.models:
            p = m.predict_proba(X)[:, 1]
            probs.append(p)
        probs = np.vstack(probs)  # [n_models, n_samples]
        avg = np.average(probs, axis=0, weights=self.weights)
        # return as 2-col proba (neg, pos) to be sklearn-compatible
        return np.vstack([1 - avg, avg]).T


def run(args):
    ensure_dirs()
    df = load_data(args.data)
    X, y, feature_names = select_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    # NEW: split train into model-train and calibration
    X_tr_model, X_cal, y_tr_model, y_cal = train_test_split(
    X_train, y_train, test_size=0.3, random_state=RANDOM_STATE, stratify=y_train
)
    # # Optional SMOTE on train only
    # X_tr, y_tr = maybe_smote(X_train, y_train, args.smote)

    # base = build_base_models()

    # calibrated_models = {}
    # for name, (model, params) in base.items():
    #     print(f"\n🔧 Training {name} ...")
    #     model.fit(X_tr, y_tr)

    #     # calibrate on original (non-resampled) validation split to avoid leakage
    #     calib, method, saved_path = calibrate_model(name, model, X_test, y_test)

    #     # evaluate raw (uncalibrated) probs and calibrated probs
    #     raw_probs = model.predict_proba(X_test)[:, 1]
    #     cal_probs = calib.predict_proba(X_test)[:, 1]

    #     # pick best threshold on calibrated probs
    #     thr, f1_at_thr = best_threshold_for_f1(y_test, cal_probs, min_pos=1)
    #     y_pred = (cal_probs >= thr).astype(int)
    #     acc = accuracy_score(y_test, y_pred)
    #     f1 = f1_score(y_test, y_pred, zero_division=0)

    #     print(f"   → {name} calibrated={method} | thr={thr:.2f} | acc={acc:.4f} | f1={f1:.4f}")
    #     print("   Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    #     # log
    #     extras = {
    #         "threshold": round(thr, 4),
    #         "calibration": method,
    #         "calibrated_model_path": saved_path,
    #         "class_counts_test": {str(k): int(v) for k, v in dict(pd.Series(y_test).value_counts()).items()},
    #         "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    #         "feature_count": len(feature_names),
    #     }
    #     params_to_log = dict(params)
    #     params_to_log.update({"smote": bool(args.smote)})
    #     log_model_result(f"{name}_calibrated", acc, f1, params=params_to_log, extra=extras, results_path=RESULTS_JSON)

    #     calibrated_models[name] = calib
    
    # SMOTE only on model-train
    X_tr, y_tr = maybe_smote(X_tr_model, y_tr_model, args.smote)

    base = build_base_models()

    calibrated_models = {}
    for name, (model, params) in base.items():
        print(f"\n🔧 Training {name} ...")
        model.fit(X_tr, y_tr)

        # Calibrate on calibration split (no SMOTE), prefit=True
        calib, method, saved_path = calibrate_model_prefit(name, model, X_cal, y_cal)

        # Use calibrated probs if available; else raw
        if calib is not None:
            probs = calib.predict_proba(X_test)[:, 1]
        else:
            probs = model.predict_proba(X_test)[:, 1]

        thr, _ = best_threshold_for_f1(y_test, probs, min_pos=1)
        y_pred = (probs >= thr).astype(int)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"   → {name} calibrated={method} | thr={thr:.2f} | acc={acc:.4f} | f1={f1:.4f}")
        print("   Confusion matrix:\n", confusion_matrix(y_test, y_pred))

        extras = {
            "threshold": round(thr, 4),
            "calibration": method,
            "calibrated_model_path": saved_path,
            "class_counts_test": {str(k): int(v) for k, v in dict(pd.Series(y_test).value_counts()).items()},
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "feature_count": X.shape[1],
        }
        params_to_log = dict(params)
        params_to_log.update({"smote": bool(args.smote)})
        log_model_result(f"{name}_calibrated", acc, f1, params=params_to_log, extra=extras, results_path=RESULTS_JSON)

        calibrated_models[name] = calib if calib is not None else model
    
    
    

    # ---- Tiny Soft-Vote Ensemble ----
    ensemble_list = []
    weights = None
    # Select available calibrated models in a stable order
    for key in ["logistic_roster", "xgboost_roster", "random_forest_roster"]:
        if key in calibrated_models:
            ensemble_list.append(calibrated_models[key])

    if not ensemble_list:
        print("\n⚠️ No calibrated models available; skipping ensemble.")
        return

    if args.weights:
        if len(args.weights) != len(ensemble_list):
            print(f"⚠️ Provided {len(args.weights)} weights for {len(ensemble_list)} models; ignoring custom weights.")
            weights = None
        else:
            weights = np.array(args.weights, dtype=float)

    ens = SoftVoteEnsemble(ensemble_list, weights=weights)
    ens_probs = ens.predict_proba(X_test)[:, 1]
    thr_ens, f1_ens = best_threshold_for_f1(y_test, ens_probs, min_pos=1)
    y_pred_ens = (ens_probs >= thr_ens).astype(int)
    acc_ens = accuracy_score(y_test, y_pred_ens)
    print(f"\n🤝 Soft‑vote ensemble | thr={thr_ens:.2f} | acc={acc_ens:.4f} | f1={f1_ens:.4f}")
    print("   Confusion matrix:\n", confusion_matrix(y_test, y_pred_ens))

    # Save ensemble wrapper
    import joblib
    joblib.dump({"ensemble": ens, "feature_names": feature_names}, BEST_ENSEMBLE_OUT)

    params_ens = {
        "members": [k for k in ["logistic_roster", "xgboost_roster", "random_forest_roster"] if k in calibrated_models],
        "weights": (weights.tolist() if weights is not None else "uniform"),
        "smote": bool(args.smote)
    }
    extras_ens = {
        "threshold": round(thr_ens, 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred_ens).tolist(),
        "class_counts_test": {str(k): int(v) for k, v in dict(pd.Series(y_test).value_counts()).items()},
        "saved_path": BEST_ENSEMBLE_OUT,
        "feature_count": len(feature_names)
    }
    log_model_result("soft_vote_ensemble_calibrated", acc_ens, f1_ens, params=params_ens, extra=extras_ens, results_path=RESULTS_JSON)


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate probabilities and build a tiny soft‑vote ensemble for roster models.")
    p.add_argument("--data", default="data/processed/roster_training_set.csv", help="Path to Part‑2 roster training CSV")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--smote", action="store_true", help="Apply SMOTE on the training split")
    p.add_argument("--weights", type=float, nargs="*", help="Optional weights for [logistic, xgboost, random_forest] (order-sensitive)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)