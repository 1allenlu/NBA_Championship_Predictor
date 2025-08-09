import argparse
import glob
import json
import os
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# our logger for part 2
from evaluate_and_log_rosters import log_model_result

# ------------------ Config Defaults ------------------
DEFAULT_DATA_GLOB = "data/processed/roster_training_set.csv"
RESULTS_JSON = "results/model_results_part2.json"
MODEL_OUT = "models/roster_pytorch_model.pkl"
TARGET = "won_championship"
NON_FEATURES = ["TEAM_NAME", "season", TARGET]
RANDOM_STATE = 42
# -----------------------------------------------------


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
    X = df.drop(columns=[c for c in NON_FEATURES if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy().fillna(0)
    y = df[TARGET].astype(int).values
    feat_names = list(X.columns)
    return X.values, y, feat_names


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers, dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)  # logits


def _metric_at_threshold(y_true, y_prob, thr, metric="f1"):
    y_pred = (y_prob >= thr).astype(int)
    if metric == "f1":
        return f1_score(y_true, y_pred, zero_division=0)
    if metric == "f0.5":
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, beta=0.5, average="binary", zero_division=0
        )
        return f
    if metric == "f2":
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, beta=2.0, average="binary", zero_division=0
        )
        return f
    if metric == "youden":
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return tpr - fpr
    return 0.0


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    return running / len(loader.dataset)


def predict_proba(model, X, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch roster-level model (Part 2).")

    # data / logging
    parser.add_argument("--data", default=DEFAULT_DATA_GLOB, help="CSV or glob for roster features")
    parser.add_argument("--results_json", default=RESULTS_JSON)
    parser.add_argument("--model_out", default=MODEL_OUT)
    parser.add_argument("--name", default="pytorch_roster", help="Base key used when logging results")

    # splits
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2, help="Portion of train used for validation")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--smote", action="store_true", help="Apply SMOTE to training split only")

    # model
    parser.add_argument("--hidden", default="128,64", help="Comma sep hidden sizes, e.g. 256,128,32")
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["Adam", "AdamW"], default="Adam")

    # threshold tuning
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--auto_threshold", action="store_true", help="Tune threshold on val set")
    parser.add_argument("--metric", default="f1", choices=["f1", "f0.5", "f2", "youden"])

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ensure_dirs()
    df = load_frames(args.data)
    print("📊 Class distribution:\n", df[TARGET].value_counts())

    X_all, y_all, feature_names = select_features(df)

    # split test first
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=args.test_size, random_state=args.seed, stratify=y_all
    )

    # from train -> make validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=args.val_size, random_state=args.seed, stratify=y_train
    )

    # SMOTE only on training subset
    if args.smote:
        sm = SMOTE(random_state=args.seed)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

    # scale (fit on train-only), save scaler
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # tensors/loaders
    train_ds = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # model
    hidden_layers = [int(x) for x in args.hidden.split(",") if x.strip()]
    model = MLP(input_dim=X_tr_s.shape[1], hidden_layers=hidden_layers, dropout=args.dropout)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # loss: if no SMOTE, use pos_weight from class imbalance; if SMOTE, use 1.0
    pos = (y_tr == 1).sum()
    neg = (y_tr == 0).sum()
    pos_weight = torch.tensor([(neg / max(pos, 1)) if not args.smote else 1.0], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # optimizer
    if args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # train
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch:>3d}/{args.epochs}  |  train_loss={train_loss:.4f}")

    # tune threshold on val (no leakage)
    best_thr = args.threshold
    if args.auto_threshold:
        val_probs = predict_proba(model, X_val_s, device)
        grid = np.linspace(0.05, 0.95, 19)
        scores = [_metric_at_threshold(y_val, val_probs, t, metric=args.metric) for t in grid]
        best_thr = float(grid[int(np.argmax(scores))])
        print(f"🎯 Auto-threshold selected: {best_thr:.2f} optimizing {args.metric.upper()} "
              f"(score={max(scores):.4f})")

    # evaluate on test with tuned threshold
    test_probs = predict_proba(model, X_test_s, device)
    test_pred = (test_probs >= best_thr).astype(int)

    acc = accuracy_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred, average="binary", zero_division=0)
    prec, rec, _, _ = precision_recall_fscore_support(
        y_test, test_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, test_pred).tolist()
    class_counts = {str(k): int(v) for k, v in pd.Series(y_test).value_counts().to_dict().items()}

    print(f"\n✅ Accuracy: {acc:.4f} | F1(+): {f1:.4f} | Prec(+): {prec:.4f} | Rec(+): {rec:.4f}")
    print("Confusion matrix:\n", np.array(cm))

    # ---- log results ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_key = f"{args.name}_{timestamp}"  # keep all runs instead of overwriting

    params = {
        "hidden_layers": hidden_layers,
        "dropout": args.dropout,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "smote": args.smote
    }
    extras = {
        "precision_pos": round(float(prec), 4),
        "recall_pos": round(float(rec), 4),
        "threshold": round(float(best_thr), 3),
        "confusion_matrix": cm,
        "class_counts_test": class_counts,
        "architecture": [X_tr_s.shape[1]] + hidden_layers + [1],
        "pos_weight": round(float(pos_weight.item()), 3)
    }
    log_model_result(model_key, acc, f1, params=params, extra=extras, results_path=args.results_json)

    # ---- save model payload for inference ----
    payload = {
        "state_dict": model.state_dict(),
        "input_dim": X_tr_s.shape[1],
        "hidden_layers": hidden_layers,
        "dropout": args.dropout,
        "scaler_mean_": scaler.mean_.tolist(),
        "scaler_scale_": scaler.scale_.tolist(),
        "threshold": float(best_thr),
        "feature_names": feature_names,
        "optimizer": args.optimizer
    }
    joblib.dump(payload, args.model_out)
    print(f"💾 Saved model package → {args.model_out}")
    print(f"📝 Logged metrics → {args.results_json}")


if __name__ == "__main__":
    main()