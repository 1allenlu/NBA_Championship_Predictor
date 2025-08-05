import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from evaluate_and_log import log_model_result

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Load and split dataset
df = pd.read_csv("data/processed/final_dataset_with_labels.csv")
X = df.drop(columns=["TEAM_NAME", "season", "won_championship"])
y = df["won_championship"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Convert to PyTorch tensors
X_res_tensor = torch.tensor(X_resampled, dtype=torch.float32)
y_res_tensor = torch.tensor(y_resampled.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

# Define PyTorch model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout=0.3):
        super(MLP, self).__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Define hyperparameters to test
configs = [
    {
        "hidden_layers": [64, 32],
        "dropout": 0.3,
        "lr": 0.001,
        "epochs": 100,
        "optimizer": "Adam"
    },
    {
        "hidden_layers": [128, 64, 32],
        "dropout": 0.2,
        "lr": 0.0005,
        "epochs": 150,
        "optimizer": "Adam"
    },
    {
        "hidden_layers": [64, 64],
        "dropout": 0.4,
        "lr": 0.001,
        "epochs": 120,
        "optimizer": "AdamW"
    }
]

# Training loop
for i, config in enumerate(configs):
    print(f"\n🔧 Training config: {config}")

    model = MLP(input_dim=X.shape[1], hidden_layers=config["hidden_layers"], dropout=config["dropout"])
    criterion = nn.BCELoss()
    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_res_tensor)
        loss = criterion(outputs, y_res_tensor)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor)
        preds_label = (preds >= 0.5).float()
        acc = accuracy_score(y_test_tensor, preds_label)
        f1 = f1_score(y_test_tensor, preds_label)

    print(f"✅ Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

    # Log
    log_model_result(
        # model_name="pytorch_nn_smote_tuned",
        model_name = f"pytorch_nn_smote_tuned_{i+1}",  # i is the index of the config
        accuracy=acc,
        f1_score=f1,
        params={
            "resampling": "SMOTE",
            "architecture": [X.shape[1]] + config["hidden_layers"] + [1],
            "dropout": config["dropout"],
            "optimizer": config["optimizer"],
            "lr": config["lr"],
            "epochs": config["epochs"]
        }
    )