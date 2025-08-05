# src/train_pytorch_model.py

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from evaluate_and_log import log_model_result
import numpy as np

# 🔧 Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🧹 Load & prepare data
df = pd.read_csv("data/processed/final_dataset_with_labels.csv")
df = df.dropna(subset=["won_championship"])

X = df.drop(columns=["TEAM_ID", "TEAM_NAME", "season", "won_championship"])
y = df["won_championship"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)

# 🧠 Define neural network
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

model = Net(X_train.shape[1]).to(device)

# ⚙️ Training setup
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100

# 🏋️‍♂️ Train model
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

# 🔍 Evaluate
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test_tensor).cpu().numpy()
    y_pred = (y_pred_probs > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 📦 Log results
log_model_result("pytorch_nn", acc, f1, {
    "architecture": [X_train.shape[1], 64, 32, 1],
    "dropout": 0.3,
    "optimizer": "Adam",
    "lr": 0.001,
    "epochs": epochs
})

# 💾 Save model
torch.save(model.state_dict(), "models/pytorch_nn_model.pth")

print(f"✅ PyTorch NN trained. Accuracy: {acc:.4f}, F1-score: {f1:.4f}")