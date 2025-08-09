# src/user_team_simulator/ensembles.py
import numpy as np

class SoftVoteEnsemble:
    """Tiny soft-vote ensemble over calibrated models."""
    def __init__(self, models, weights=None):
        self.models = models
        if weights is None:
            self.weights = np.ones(len(models)) / len(models)
        else:
            w = np.array(weights, dtype=float)
            self.weights = w / w.sum()

    def predict_proba(self, X):
        probs = []
        for m in self.models:
            p = m.predict_proba(X)[:, 1]
            probs.append(p)
        probs = np.vstack(probs)  # [n_models, n_samples]
        avg = np.average(probs, axis=0, weights=self.weights)
        return np.vstack([1 - avg, avg]).T