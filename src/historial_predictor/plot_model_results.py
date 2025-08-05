import json
import matplotlib.pyplot as plt
import os

# Path to the results file
results_path = "results/model_results.json"

# Load results
if not os.path.exists(results_path):
    raise FileNotFoundError(f"❌ Could not find {results_path}. Make sure it exists.")

with open(results_path, "r") as f:
    results = json.load(f)

# Sort models by F1 score (descending)
sorted_items = sorted(results.items(), key=lambda item: item[1]["f1_score"], reverse=True)

models = [name for name, _ in sorted_items]
accuracies = [res["accuracy"] for _, res in sorted_items]
f1_scores = [res["f1_score"] for _, res in sorted_items]

# Create plot
plt.figure(figsize=(14, 6))
bar_width = 0.35
x = range(len(models))

plt.bar(x, accuracies, width=bar_width, label="Accuracy", alpha=0.8)
plt.bar([i + bar_width for i in x], f1_scores, width=bar_width, label="F1 Score", alpha=0.8)

plt.xlabel("Models")
plt.ylabel("Score")
plt.title("Model Performance (sorted by F1 Score)")
plt.xticks([i + bar_width / 2 for i in x], models, rotation=45, ha="right")
plt.legend()
plt.tight_layout()

# Save and display
os.makedirs("results", exist_ok=True)
plt.savefig("results/model_performance_plot.png")
plt.show()