# src/user_team_simulator/cv_roster_models.py
import argparse, json, os, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from evaluate_and_log_rosters import log_model_result

RANDOM_STATE = 42
RESULTS_JSON = "results/result_part2.json"
TARGET = "won_championship"
DROP = ["TEAM_NAME","season"]

def threshold_sweep(y_true, y_prob, grid=None):
    if grid is None: grid = np.linspace(0.05, 0.95, 19)
    best = (0.5, 0.0)  # (thr, f1)
    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best[1]: best = (float(t), float(f1))
    return best

def run_model(name, model, X, y, n_splits=5, use_smote=True):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    f1s, accs = [], []
    oof_prob = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        if use_smote:
            sm = SMOTE(random_state=RANDOM_STATE)
            X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:,1]
        oof_prob[te] = prob
        thr, _ = threshold_sweep(y_te, prob)
        pred = (prob >= thr).astype(int)
        f1s.append(f1_score(y_te, pred, zero_division=0))
        accs.append(accuracy_score(y_te, pred))
    thr_all, f1_at_thr = threshold_sweep(y, oof_prob)
    return {
        "cv_f1_mean": float(np.mean(f1s)), "cv_f1_std": float(np.std(f1s)),
        "cv_acc_mean": float(np.mean(accs)), "cv_acc_std": float(np.std(accs)),
        "recommended_threshold": thr_all, "oof_f1_at_recommended_threshold": float(f1_at_thr)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/roster_training_set.csv")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--smote", action="store_true")
    args = ap.parse_args()

    Path("results").mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(args.data)
    X = df.drop(columns=[c for c in DROP if c in df.columns] + [TARGET]).select_dtypes(include=[np.number]).fillna(0)
    y = df[TARGET].astype(int)

    # 1) Logistic (with scaling)
    log_reg = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    out = run_model("logistic_roster_cv", log_reg, X, y, n_splits=args.splits, use_smote=args.smote)
    log_model_result("logistic_roster_cv", out["cv_acc_mean"], out["cv_f1_mean"],
                     params={"cv_splits": args.splits, "smote": args.smote},
                     extra=out, results_path=RESULTS_JSON)

    # 2) Random Forest
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    out = run_model("random_forest_roster_cv", rf, X, y, n_splits=args.splits, use_smote=args.smote)
    log_model_result("random_forest_roster_cv", out["cv_acc_mean"], out["cv_f1_mean"],
                     params={"n_estimators": 500, "min_samples_leaf": 2, "smote": args.smote},
                     extra=out, results_path=RESULTS_JSON)

    # 3) XGBoost
    xgb = XGBClassifier(
        n_estimators=700, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
        min_child_weight=1, eval_metric="logloss", tree_method="hist",
        random_state=RANDOM_STATE
    )
    out = run_model("xgboost_roster_cv", xgb, X, y, n_splits=args.splits, use_smote=args.smote)
    log_model_result("xgboost_roster_cv", out["cv_acc_mean"], out["cv_f1_mean"],
                     params={"n_estimators": 700, "max_depth": 3, "smote": args.smote},
                     extra=out, results_path=RESULTS_JSON)

if __name__ == "__main__":
    main()