import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from evaluate_and_log import log_model_result

df = pd.read_csv("data/processed/final_dataset_with_labels.csv")
df = df.dropna(subset=["won_championship"])

X = df.drop(columns=["won_championship", "TEAM_NAME", "season"])
y = df["won_championship"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
# log_model_result("xgboost", acc, f1)
log_model_result("xgboost", acc, f1, {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "eval_metric": "logloss"
})