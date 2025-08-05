import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
data_path = "data/processed/final_dataset.csv"
df = pd.read_csv(data_path)

# Prepare features & target
X = df.drop(columns=["W", "TEAM_NAME", "season"])  # "W" is the target (Wins)
y = df["W"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Option A: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Option B: Logistic Regression (optional)
# lr = LogisticRegression(max_iter=1000)
# lr.fit(X_train, y_train)
# y_pred_lr = lr.predict(X_test)

# Evaluation
print("📊 Random Forest Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/random_forest_model.pkl")
print("✅ Model saved to models/random_forest_model.pkl")