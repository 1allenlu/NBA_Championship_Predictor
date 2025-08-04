import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import json
from evaluate_and_log import log_model_result

# Load your labeled data
df = pd.read_csv("data/processed/final_dataset_with_labels.csv")

# Features and target
X = df.drop(columns=["won_championship", "TEAM_NAME", "season"])
y = df["won_championship"]

print("📊 Full dataset class distribution:")
print(y.value_counts())

# Stratified split to preserve class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n📊 Training set class distribution:")
print(y_train.value_counts())

print("\n📊 Test set class distribution:")
print(y_test.value_counts())

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("\n✅ After SMOTE:")
print(y_resampled.value_counts())

# Calculate scale_pos_weight manually for reference (not used here since we do SMOTE)
scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
print(f"\n🔢 scale_pos_weight (for reference): {scale_pos_weight:.2f}")


# Initialize XGBoost with tuned params + scale_pos_weight
model = XGBClassifier(
    learning_rate=0.2,
    max_depth=3,
    n_estimators=200,
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)


# # Initialize XGBoost with tuned parameters
# model = XGBClassifier(
#     learning_rate=0.2,
#     max_depth=3,
#     n_estimators=200,
#     use_label_encoder=False,
#     eval_metric="logloss",  # Could also try 'auc'
#     random_state=42
# )

# Train
model.fit(X_resampled, y_resampled)

# Predict
# y_pred = model.predict(X_test)
# Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# Set custom threshold
threshold = 0.3  # Try 0.3, or lower later
y_pred = (y_probs >= threshold).astype(int)



import numpy as np
print("🔍 Predicted class counts:", np.bincount(y_pred))
print("🎯 True class counts:", np.bincount(y_test))




# Analyze predictions
print("\n🔍 Predicted label counts:", np.bincount(y_pred))

# Evaluate
acc = accuracy_score(y_test, y_pred)

print("✅ Unique values in predictions:", np.unique(y_pred, return_counts=True))
print("✅ Unique values in actual labels:", np.unique(y_test, return_counts=True))


from sklearn.metrics import classification_report

print("\n🧾 Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("✅ Predictions:", np.bincount(y_pred))
print("✅ Ground truth:", np.bincount(y_test))


f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)

print(f"\n✅ Accuracy: {acc:.4f}")
print(f"✅ F1 Score: {f1:.4f}")

# Log results
log_model_result(
    "xgboost_smote_tuned_thresh_0.3", acc, f1,
    params={
        "resampling": "SMOTE",
        "learning_rate": 0.2,
        "max_depth": 3,
        "n_estimators": 200,
        "scale_pos_weight": scale_pos_weight,
        "threshold": 0.3
    }
)