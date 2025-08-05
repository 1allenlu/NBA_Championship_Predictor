import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from evaluate_and_log import log_model_result

# Load dataset
df = pd.read_csv("data/processed/final_dataset_with_labels.csv")

# Features and target
X = df.drop(columns=["TEAM_ID", "TEAM_NAME", "season", "won_championship"])
y = df["won_championship"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_res, y_train_res)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Log results
log_model_result("random_forest_smote", acc, f1, params={
    "resampling": "SMOTE",
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
})

print(f"✅ Random Forest (SMOTE) -> Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")