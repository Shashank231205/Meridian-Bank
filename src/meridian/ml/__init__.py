"""Machine learning from scratch: logistic regression, k-means, metrics."""

from .kmeans import KMeansFit, choose_k, fit_kmeans, silhouette_score
from .logistic import LogisticFit, fit_logistic, standardise
from .metrics import (
    ConfusionMatrix,
    average_precision,
    brier_score,
    confusion_matrix,
    evaluate,
    ks_statistic,
    lift_table,
    optimal_threshold,
    roc_auc,
    roc_curve,
)

__all__ = [
    "LogisticFit", "fit_logistic", "standardise",
    "KMeansFit", "fit_kmeans", "choose_k", "silhouette_score",
    "ConfusionMatrix", "confusion_matrix", "roc_auc", "roc_curve",
    "average_precision", "brier_score", "ks_statistic", "lift_table",
    "optimal_threshold", "evaluate",
]
