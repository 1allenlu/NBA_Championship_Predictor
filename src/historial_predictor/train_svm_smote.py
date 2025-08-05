import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from evaluate_and_log import log_model_result

# Load data
df = pd.read_csv("data/processed/final_dataset_with_labels.csv")
X = df.drop(columns=["won_championship", "TEAM_NAME", "season"])
y = df["won_championship"]

# Resample
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Train
model = SVC(kernel="rbf", class_weight="balanced")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Log
log_model_result("svm_smote", acc, f1, {
    "resampling": "SMOTE",
    "kernel": "rbf",
    "class_weight": "balanced"
})