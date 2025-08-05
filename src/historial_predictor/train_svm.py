# src/train_svm.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from evaluate_and_log import log_model_result

# 🔹 Load data
df = pd.read_csv("data/processed/final_dataset_with_labels.csv")

# 🔹 Drop rows with missing labels (just in case)
df = df.dropna(subset=["won_championship"])

# 🔹 Features and target
X = df.drop(columns=["TEAM_ID", "TEAM_NAME", "season", "won_championship"])
y = df["won_championship"]

# 🔹 Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 🔹 Define pipeline with StandardScaler (important for SVMs)
model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", class_weight="balanced", probability=False)
)

# 🔹 Train model
model.fit(X_train, y_train)

# 🔹 Predict & evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 🔹 Log results
log_model_result("svm", acc, f1, {
    "kernel": "rbf",
    "class_weight": "balanced"
})

print(f"✅ SVM trained. Accuracy: {acc:.4f}, F1-score: {f1:.4f}")