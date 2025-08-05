import json
import os

def log_model_result(model_name, accuracy, f1_score, params=None):
    result_file = "results/model_results.json"
    
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    data[model_name] = {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1_score, 4),
        "params": params or {}
    }

    with open(result_file, "w") as f:
        json.dump(data, f, indent=4)