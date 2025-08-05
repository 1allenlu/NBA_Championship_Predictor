import pandas as pd
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data/processed/final_dataset_with_labels.csv")
X = df.drop(columns=["won_championship", "TEAM_NAME", "season"])
y = df["won_championship"]

# Train/test split
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# SMOTE
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Train model
model = XGBClassifier(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=100,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_resampled, y_resampled)

# Save model
joblib.dump(model, "models/best_model.pkl")
print("✅ Model saved to models/best_model.pkl")