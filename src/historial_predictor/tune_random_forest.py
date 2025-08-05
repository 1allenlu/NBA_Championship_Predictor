import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from evaluate_and_log import log_model_result

# Load and prepare data
df = pd.read_csv("data/processed/final_dataset_with_labels.csv")
X = df.drop(columns=["won_championship", "TEAM_NAME", "season"])
y = df["won_championship"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Define hyperparameter grid
param_grid = [
    {"n_estimators": 50, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1},
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 2},
    {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5, "min_samples_leaf": 4},
    {"n_estimators": 150, "max_depth": 8, "min_samples_split": 3, "min_samples_leaf": 2},
]

for i, params in enumerate(param_grid):
    print(f"\n🔧 Config {i+1}: {params}")
    model = RandomForestClassifier(**params, random_state=42)

    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"✅ Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

    log_model_result(
        f"random_forest_smote_tuned_{i+1}", acc, f1,
        params={**params, "resampling": "SMOTE"}
    )