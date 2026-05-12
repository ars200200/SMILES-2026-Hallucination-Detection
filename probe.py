"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module`` for compatibility with the evaluation harness.
    The fitted classifier is a regularized linear probe over selected,
    standardized hidden-state features.
    """

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None  # built lazily in fit()
        self._scaler = StandardScaler()
        self._threshold: float = 0.5  # tuned by fit_hyperparameters()
        self._models: list = []

    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the network definition below.
    # ------------------------------------------------------------------
    def _build_network(self, input_dim: int) -> None:
        """Instantiate the network layers.

        Called once at the start of ``fit()`` when ``input_dim`` is known.

        Args:
            input_dim: Feature vector dimensionality.
        """
        hidden_dim = min(256, max(32, input_dim // 8))
        self._net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,)``.

        Args:
            x: Float tensor of shape ``(n_samples, feature_dim)``.

        Returns:
            1-D tensor of raw (pre-sigmoid) logits.
        """
        if self._net is None and self._models:
            x_np = x.detach().cpu().numpy()
            prob_pos = self.predict_proba(x_np)[:, 1]
            prob_pos = np.clip(prob_pos, 1e-6, 1.0 - 1e-6)
            logits = np.log(prob_pos / (1.0 - prob_pos))
            return torch.from_numpy(logits).to(device=x.device, dtype=x.dtype)
        if self._net is None:
            raise RuntimeError("Probe has not been fitted yet. Call fit() first.")
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Selects the strongest univariate features, scales them, and fits a
        class-balanced logistic regression probe.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        self._models = []
        feature_count = min(2048, X.shape[1])
        for c_value in (0.003,):
            model = make_pipeline(
                StandardScaler(),
                SelectKBest(f_classif, k=feature_count),
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=42,
                    solver="liblinear",
                ),
            )
            model.fit(np.asarray(X, dtype=np.float32), y.astype(int))
            self._models.append(model)

        self._threshold = 0.5
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise accuracy.

        The chosen threshold is stored in ``self._threshold`` and used by
        subsequent ``predict`` calls.  Call this after ``fit`` and before
        ``predict``.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        probs = self.predict_proba(X_val)[:, 1]

        # Candidate thresholds: unique predicted probabilities plus a coarse grid.
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold = 0.5
        best_accuracy = -1.0
        best_f1 = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            acc = accuracy_score(y_val, y_pred_t)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if acc > best_accuracy or (acc == best_accuracy and score > best_f1):
                best_accuracy = acc
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        if not self._models:
            raise RuntimeError("Probe has not been fitted yet. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float32)
        probabilities = [model.predict_proba(X_arr)[:, 1] for model in self._models]
        prob_pos = np.mean(probabilities, axis=0)
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

