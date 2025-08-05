import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from evaluate_and_log import log_model_result

# Load dataset
df = pd.read_csv("data/processed/final_dataset_with_labels.csv")
X = df.drop(columns=["won_championship", "TEAM_NAME", "season"])
y = df["won_championship"]

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test_np = y_test.values  # for evaluation

# 🧠 Define Neural Network
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.drop2 = nn.Dropout(0.3)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop1(x)
        x = torch.relu(self.fc2(x))
        x = self.drop2(x)
        return torch.sigmoid(self.out(x))

# Setup
model = Net(input_dim=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100

# Training loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_label = (y_pred > 0.5).int().numpy()
    acc = accuracy_score(y_test_np, y_pred_label)
    f1 = f1_score(y_test_np, y_pred_label)

# Log results
log_model_result("pytorch_nn_smote", acc, f1, {
    "resampling": "SMOTE",
    "architecture": [X_train.shape[1], 64, 32, 1],
    "dropout": 0.3,
    "optimizer": "Adam",
    "lr": 0.001,
    "epochs": epochs
})

print(f"✅ PyTorch model with SMOTE — Acc: {acc:.4f}, F1: {f1:.4f}")