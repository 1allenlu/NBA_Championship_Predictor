# src/user_team_simulator/evaluate_and_log_rosters.py
import json, os, time
import numpy as np

RESULTS_PATH = "results/model_results_part2.json"

def _to_py(obj):
    if isinstance(obj, dict):
        return { _to_py(k): _to_py(v) for k, v in obj.items() }
    if isinstance(obj, (list, tuple)):
        return [ _to_py(v) for v in obj ]
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)):    return bool(obj)
    if isinstance(obj, (np.ndarray,)):  return obj.tolist()
    return obj

def log_model_result(model_name: str, accuracy: float, f1: float, params: dict = None,
                     extra: dict | None = None, results_path: str = RESULTS_PATH):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # load existing
    data = {}
    if os.path.exists(results_path) and os.path.getsize(results_path) > 0:
        try:
            with open(results_path, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}

    # ensure list history per model
    if model_name not in data:
        data[model_name] = []

    payload = {
        "accuracy": round(float(accuracy), 4),
        "f1_score": round(float(f1), 4),
        "params": params or {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        payload.update(_to_py(extra))

    data[model_name].append(_to_py(payload))

    with open(results_path, "w") as f:
        json.dump(data, f, indent=4)