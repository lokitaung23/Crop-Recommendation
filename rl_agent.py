from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.utils.validation import check_is_fitted
import pickle

@dataclass
class ContextualBanditAgent:
    n_actions: int
    feature_dim: int
    epsilon: float = 0.10
    learning_rate: float = 0.01
    random_state: int = 42
    models: List[SGDRegressor] = field(default_factory=list)
    action_counts: np.ndarray = field(default_factory=lambda: np.zeros(0))
    action_rewards: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self):
        if not self.models:
            rng = np.random.RandomState(self.random_state)
            for _ in range(self.n_actions):
                self.models.append(
                    SGDRegressor(loss="squared_error", learning_rate="optimal", random_state=rng)
                )
        if self.action_counts.size == 0:
            self.action_counts = np.zeros(self.n_actions, dtype=np.int64)
        if self.action_rewards.size == 0:
            self.action_rewards = np.zeros(self.n_actions, dtype=np.float64)

    def _ensure_shape(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        return x

    def predict_reward(self, X: np.ndarray) -> np.ndarray:
        X = self._ensure_shape(X)
        preds = np.zeros((X.shape[0], self.n_actions), dtype=np.float64)
        for a, m in enumerate(self.models):
            try:
                check_is_fitted(m)
                preds[:, a] = m.predict(X)
            except Exception:
                preds[:, a] = 0.5  # optimistic prior
        return preds

    def choose_action(self, X: np.ndarray, base_policy_proba: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        X = self._ensure_shape(X)
        n = X.shape[0]
        reward_preds = self.predict_reward(X)
        if base_policy_proba is not None:
            rp = reward_preds.copy()
            rp_min = rp.min(axis=1, keepdims=True)
            rp_max = rp.max(axis=1, keepdims=True)
            denom = np.where((rp_max - rp_min) == 0, 1.0, (rp_max - rp_min))
            rp_norm = (rp - rp_min) / denom
            scores = 0.6 * rp_norm + 0.4 * base_policy_proba
        else:
            scores = reward_preds

        best_actions = scores.argmax(axis=1)
        actions, chosen_scores = [], []
        rng = np.random.RandomState(self.random_state)
        for i in range(n):
            a = rng.randint(0, self.n_actions) if rng.rand() < self.epsilon else int(best_actions[i])
            actions.append(a)
            chosen_scores.append(scores[i, a])
        return np.array(actions, dtype=int), np.array(chosen_scores, dtype=np.float64)

    def update(self, X: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        X = self._ensure_shape(X)
        actions = np.asarray(actions, dtype=int)
        rewards = np.asarray(rewards, dtype=np.float64)
        assert X.shape[0] == actions.shape[0] == rewards.shape[0]
        for a in range(self.n_actions):
            idx = np.where(actions == a)[0]
            if idx.size == 0:
                continue
            Xa, ra = X[idx], rewards[idx]
            self.models[a].partial_fit(Xa, ra)
            self.action_counts[a] += idx.size
            self.action_rewards[a] += ra.sum()

    def stats(self) -> Dict[str, Any]:
        avg = np.divide(self.action_rewards, np.where(self.action_counts == 0, 1, self.action_counts))
        return {"counts": self.action_counts.tolist(), "sum_rewards": self.action_rewards.tolist(), "avg_rewards": avg.tolist()}

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ContextualBanditAgent":
        with open(path, "rb") as f:
            return pickle.load(f)
