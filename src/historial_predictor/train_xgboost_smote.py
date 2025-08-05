import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score
from evaluate_and_log import log_model_result

# Load data
df = pd.read_csv("data/processed/final_dataset_with_labels.csv")
X = df.drop(columns=["won_championship", "TEAM_NAME", "season"])
y = df["won_championship"]

# SMOTE resampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Train XGBoost
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    eval_metric="logloss",
    use_label_encoder=False
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Log
log_model_result("xgboost_smote", acc, f1, {
    "resampling": "SMOTE",
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "eval_metric": "logloss"
})