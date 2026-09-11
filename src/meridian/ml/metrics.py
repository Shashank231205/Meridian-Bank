"""Classification metrics, implemented from scratch.

Accuracy is deliberately not the headline. At an 11% conversion rate or a 36%
churn rate, a model that predicts the majority class for everyone scores 89% or
64% and is useless. The metrics that matter for imbalanced problems are AUC,
precision/recall at the threshold you will actually deploy, and lift in the top
decile -- which is the number a campaign team can act on directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ConfusionMatrix:
    """A 2x2 confusion matrix at one threshold."""

    tp: int
    fp: int
    tn: int
    fn: int
    threshold: float = 0.5

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def precision(self) -> float:
        """Of those flagged, what share were right."""
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        """Of the actual events, what share were caught. Also sensitivity."""
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def specificity(self) -> float:
        d = self.tn + self.fp
        return self.tn / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def balanced_accuracy(self) -> float:
        """Mean of recall and specificity -- honest under class imbalance."""
        return (self.recall + self.specificity) / 2

    @property
    def mcc(self) -> float:
        """Matthews correlation coefficient.

        The single most informative scalar for an imbalanced binary problem: it
        is high only when all four cells of the matrix are good, so unlike F1 it
        cannot be gamed by ignoring the negative class.
        """
        num = self.tp * self.tn - self.fp * self.fn
        den = np.sqrt(
            float((self.tp + self.fp) * (self.tp + self.fn)
                  * (self.tn + self.fp) * (self.tn + self.fn))
        )
        return float(num / den) if den > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "accuracy": round(self.accuracy, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "specificity": round(self.specificity, 6),
            "f1": round(self.f1, 6),
            "balanced_accuracy": round(self.balanced_accuracy, 6),
            "mcc": round(self.mcc, 6),
        }


def confusion_matrix(
    y_true, y_score, *, threshold: float = 0.5
) -> ConfusionMatrix:
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(y_score, dtype=float).ravel()
    pred = (p >= threshold).astype(int)
    return ConfusionMatrix(
        tp=int(((pred == 1) & (y == 1)).sum()),
        fp=int(((pred == 1) & (y == 0)).sum()),
        tn=int(((pred == 0) & (y == 0)).sum()),
        fn=int(((pred == 0) & (y == 1)).sum()),
        threshold=threshold,
    )


def roc_auc(y_true, y_score) -> float:
    """Area under the ROC curve, via the rank (Mann-Whitney) identity.

    AUC equals the probability that a randomly chosen positive outranks a
    randomly chosen negative, which is exactly the normalised Mann-Whitney U.
    Computing it by ranking is O(n log n) and exact, where trapezoidal
    integration over a sampled curve is neither.
    """
    y = np.asarray(y_true, dtype=int).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")   # undefined with only one class present

    ranks = _rank_with_ties(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _rank_with_ties(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def roc_curve(y_true, y_score) -> pd.DataFrame:
    """False and true positive rates across every distinct threshold."""
    y = np.asarray(y_true, dtype=int).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    order = np.argsort(-s)
    y_sorted = y[order]

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    n_pos, n_neg = tps[-1], fps[-1]
    if n_pos == 0 or n_neg == 0:
        return pd.DataFrame(columns=["threshold", "fpr", "tpr"])

    return pd.DataFrame({
        "threshold": s[order],
        "fpr": fps / n_neg,
        "tpr": tps / n_pos,
    })


def average_precision(y_true, y_score) -> float:
    """Area under the precision-recall curve.

    More informative than ROC AUC when positives are rare, because it ignores
    true negatives -- of which an imbalanced problem has an overwhelming number.
    """
    y = np.asarray(y_true, dtype=int).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    order = np.argsort(-s)
    y_sorted = y[order]

    tps = np.cumsum(y_sorted)
    n_pos = tps[-1]
    if n_pos == 0:
        return float("nan")

    precision = tps / np.arange(1, len(y_sorted) + 1)
    recall = tps / n_pos
    # Sum precision at each point where recall increases -- the standard
    # step-wise AP, which does not reward interpolation artefacts.
    d_recall = np.diff(np.concatenate([[0.0], recall]))
    return float((precision * d_recall).sum())


def brier_score(y_true, y_score) -> float:
    """Mean squared error of the predicted probabilities.

    A calibration measure: a model can rank perfectly (AUC 1.0) while being
    systematically over-confident, and only a calibration metric catches that.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_score, dtype=float).ravel()
    return float(np.mean((p - y) ** 2))


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov separation, the standard measure in credit risk."""
    curve = roc_curve(y_true, y_score)
    return 0.0 if curve.empty else float((curve["tpr"] - curve["fpr"]).max())


def lift_table(y_true, y_score, *, n_bins: int = 10) -> pd.DataFrame:
    """Decile lift table -- the output a campaign team can act on.

    Answers "if we contact the top 10% by score, how many more events do we
    catch than random targeting would". That is a budget decision, which is why
    this table matters more to the business than AUC does.
    """
    y = np.asarray(y_true, dtype=int).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    n = len(y)
    if n == 0:
        return pd.DataFrame()

    order = np.argsort(-s)
    y_sorted, s_sorted = y[order], s[order]
    base_rate = y.mean()

    edges = np.linspace(0, n, n_bins + 1).astype(int)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        chunk = y_sorted[lo:hi]
        rate = chunk.mean()
        cum = y_sorted[:hi]
        rows.append({
            "decile": i + 1,
            "n": hi - lo,
            "min_score": round(float(s_sorted[hi - 1]), 6),
            "max_score": round(float(s_sorted[lo]), 6),
            "n_events": int(chunk.sum()),
            "event_rate": round(float(rate), 6),
            "lift": round(float(rate / base_rate), 4) if base_rate else 0.0,
            "cumulative_events": int(cum.sum()),
            "cumulative_capture_pct": round(
                100.0 * cum.sum() / y.sum(), 2) if y.sum() else 0.0,
            "cumulative_lift": round(
                float(cum.mean() / base_rate), 4) if base_rate else 0.0,
        })
    return pd.DataFrame(rows)


def optimal_threshold(y_true, y_score, *, metric: str = "f1") -> dict[str, Any]:
    """Find the threshold maximising a chosen metric.

    0.5 is rarely the right cut for an imbalanced problem: it is the default
    only because it is the midpoint, not because it optimises anything.
    """
    y = np.asarray(y_true, dtype=int).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    candidates = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 99)))

    best = {"threshold": 0.5, "value": -np.inf, "metric": metric}
    for t in candidates:
        cm = confusion_matrix(y, s, threshold=float(t))
        value = {"f1": cm.f1, "mcc": cm.mcc,
                 "balanced_accuracy": cm.balanced_accuracy,
                 "precision": cm.precision, "recall": cm.recall}[metric]
        if value > best["value"]:
            best = {"threshold": float(t), "value": float(value), "metric": metric,
                    **cm.to_dict()}
    return best


def evaluate(y_true, y_score, *, threshold: float = 0.5) -> dict[str, Any]:
    """The full metric set for one model."""
    cm = confusion_matrix(y_true, y_score, threshold=threshold)
    lift = lift_table(y_true, y_score)
    return {
        **cm.to_dict(),
        "roc_auc": round(roc_auc(y_true, y_score), 6),
        "average_precision": round(average_precision(y_true, y_score), 6),
        "brier_score": round(brier_score(y_true, y_score), 6),
        "ks_statistic": round(ks_statistic(y_true, y_score), 6),
        "base_rate": round(float(np.mean(y_true)), 6),
        "top_decile_lift": float(lift["lift"].iloc[0]) if not lift.empty else 0.0,
        "top_decile_capture_pct": float(lift["cumulative_capture_pct"].iloc[0])
                                  if not lift.empty else 0.0,
    }
