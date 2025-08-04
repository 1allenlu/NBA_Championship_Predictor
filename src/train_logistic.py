from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib
from evaluate_and_log import log_model_result

# Load data
df = pd.read_csv("data/processed/final_dataset_with_labels.csv")
X = df.drop(columns=["TEAM_NAME", "season", "won_championship"])
y = df["won_championship"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# 🧪 Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train_resampled, y_train_resampled)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Save model + metrics
joblib.dump(model, "models/logistic_regression_smote.pkl")
log_model_result("logistic_regression_smote", acc, f1, {
    "resampling": "SMOTE",
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000
})