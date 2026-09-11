"""k-means clustering with k-means++ initialisation and model selection.

Written from scratch for the same reason as the logistic: no scipy, no sklearn.

Two details matter more than the algorithm itself.

**k-means++ initialisation.** Seeding centroids uniformly at random regularly
converges to a visibly bad local optimum -- two centroids inside one dense
cluster and none in another. k-means++ samples each new centroid with
probability proportional to squared distance from the nearest existing one,
which spreads the seeds and, in expectation, lands within a log(k) factor of the
optimal clustering. It costs one extra pass.

**Choosing k.** The elbow method alone is a judgement call dressed as a method,
and different people read different elbows off the same curve. Silhouette gives
a defensible number, so both are computed and reported together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..common.logging import get_logger

log = get_logger(__name__)


@dataclass
class KMeansFit:
    """A fitted k-means model."""

    centroids: np.ndarray            # (k, n_features)
    labels: np.ndarray               # (n,)
    inertia: float                   # within-cluster sum of squares
    n_iterations: int
    converged: bool
    k: int
    feature_names: list[str] = field(default_factory=list)
    silhouette: float | None = None

    def cluster_sizes(self) -> pd.Series:
        return pd.Series(self.labels).value_counts().sort_index()

    def centroid_frame(self) -> pd.DataFrame:
        """Centroids as a labelled frame, with cluster sizes."""
        names = self.feature_names or [f"f{i}" for i in range(self.centroids.shape[1])]
        df = pd.DataFrame(self.centroids, columns=names)
        df.insert(0, "cluster", range(self.k))
        df["size"] = self.cluster_sizes().reindex(range(self.k), fill_value=0).to_numpy()
        df["share_pct"] = (df["size"] / df["size"].sum() * 100).round(2)
        return df

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return _assign(X, self.centroids)


def fit_kmeans(
    X, k: int, *, max_iter: int = 300, tol: float = 1e-6,
    n_init: int = 10, seed: int = 20260911,
    feature_names: list[str] | None = None,
    compute_silhouette: bool = True,
    silhouette_sample: int = 5000,
) -> KMeansFit:
    """Fit k-means, keeping the best of ``n_init`` restarts.

    Multiple restarts because even k-means++ can land in a poor local optimum;
    the best-inertia run is kept.
    """
    names = list(feature_names) if feature_names is not None else (
        list(X.columns) if isinstance(X, pd.DataFrame) else []
    )
    data = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) \
        else np.asarray(X, dtype=float)

    n = len(data)
    if k < 1:
        raise ValueError("k must be at least 1")
    if k > n:
        raise ValueError(f"k={k} exceeds the {n} available points")

    rng = np.random.default_rng(seed)
    best: KMeansFit | None = None

    for _restart in range(n_init):
        centroids = _kmeanspp_init(data, k, rng)
        labels = np.zeros(n, dtype=int)
        converged = False
        iteration = 0

        # noqa on B007: the loop variable is not used inside the body, but it
        # is read after the loop to record how many iterations convergence took.
        for iteration in range(1, max_iter + 1):  # noqa: B007
            labels = _assign(data, centroids)
            new_centroids = np.empty_like(centroids)
            for j in range(k):
                members = data[labels == j]
                # An empty cluster is re-seeded at the point furthest from its
                # centroid rather than left empty, which would silently reduce k.
                if len(members) == 0:
                    far = np.argmax(_min_sq_dist(data, centroids))
                    new_centroids[j] = data[far]
                else:
                    new_centroids[j] = members.mean(axis=0)

            shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
            centroids = new_centroids
            if shift < tol:
                converged = True
                break

        inertia = float(np.sum(_min_sq_dist(data, centroids)))
        if best is None or inertia < best.inertia:
            best = KMeansFit(centroids=centroids, labels=labels, inertia=inertia,
                             n_iterations=iteration, converged=converged, k=k,
                             feature_names=names)

    assert best is not None
    if compute_silhouette and 1 < k < n:
        best.silhouette = silhouette_score(
            data, best.labels, sample_size=silhouette_sample, seed=seed
        )

    log.info("k-means k=%d: inertia=%.2f, silhouette=%s, converged=%s",
             k, best.inertia,
             f"{best.silhouette:.4f}" if best.silhouette is not None else "n/a",
             best.converged)
    return best


def _kmeanspp_init(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: spread the initial centroids by squared distance."""
    n = len(data)
    centroids = np.empty((k, data.shape[1]))
    centroids[0] = data[rng.integers(n)]

    for j in range(1, k):
        d2 = _min_sq_dist(data, centroids[:j])
        total = d2.sum()
        if total <= 0:
            # Every point coincides with a centroid: pick at random.
            centroids[j] = data[rng.integers(n)]
        else:
            centroids[j] = data[rng.choice(n, p=d2 / total)]
    return centroids


def _assign(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Nearest-centroid assignment."""
    return np.argmin(_sq_dists(data, centroids), axis=1)


def _sq_dists(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Squared euclidean distance from every point to every centroid.

    Expanded as |x|^2 - 2x.c + |c|^2 rather than materialising an
    (n, k, d) difference tensor, which would exhaust memory at 25,000 points.
    """
    return (
        np.sum(data ** 2, axis=1)[:, None]
        - 2 * data @ centroids.T
        + np.sum(centroids ** 2, axis=1)[None, :]
    )


def _min_sq_dist(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.maximum(_sq_dists(data, centroids).min(axis=1), 0.0)


def silhouette_score(
    X, labels, *, sample_size: int = 5000, seed: int = 20260911
) -> float:
    """Mean silhouette coefficient.

    For each point: (b - a) / max(a, b), where a is the mean distance to its own
    cluster and b the mean distance to the nearest other cluster. Ranges from -1
    to 1; above 0.5 is a strong structure, near 0 means overlapping clusters.

    Subsampled by default because the exact computation is O(n^2) -- at 25,000
    customers that is 625 million pairwise distances for a number that is stable
    to three decimals on a few thousand points.
    """
    data = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) \
        else np.asarray(X, dtype=float)
    lab = np.asarray(labels, dtype=int)

    n = len(data)
    if n > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=sample_size, replace=False)
        data, lab = data[idx], lab[idx]
        n = sample_size

    unique = np.unique(lab)
    if len(unique) < 2:
        return 0.0

    # Full pairwise distance matrix on the (sub)sample.
    d2 = (np.sum(data ** 2, axis=1)[:, None]
          - 2 * data @ data.T
          + np.sum(data ** 2, axis=1)[None, :])
    dist = np.sqrt(np.maximum(d2, 0))

    scores = np.zeros(n)
    for i in range(n):
        own = lab == lab[i]
        own_count = own.sum() - 1
        if own_count <= 0:
            scores[i] = 0.0     # a singleton cluster has no defined cohesion
            continue
        a = dist[i, own].sum() / own_count
        b = min(
            dist[i, lab == other].mean()
            for other in unique if other != lab[i]
        )
        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0

    return float(scores.mean())


def choose_k(
    X, k_range=range(2, 11), *, seed: int = 20260911, n_init: int = 5,
) -> pd.DataFrame:
    """Evaluate a range of k by inertia, silhouette and the elbow.

    Reports both criteria because they answer different questions and sometimes
    disagree: silhouette measures separation, the elbow measures diminishing
    returns on variance explained. Where they disagree, silhouette is the more
    defensible choice and the disagreement is worth stating.
    """
    rows = []
    for k in k_range:
        fit = fit_kmeans(X, k, seed=seed, n_init=n_init)
        rows.append({
            "k": k,
            "inertia": round(fit.inertia, 4),
            "silhouette": round(fit.silhouette, 6) if fit.silhouette is not None else None,
            "n_iterations": fit.n_iterations,
        })
    df = pd.DataFrame(rows)

    # Elbow by the maximum second difference of inertia: the point where the
    # curve stops falling steeply.
    if len(df) >= 3:
        d1 = df["inertia"].diff()
        d2 = d1.diff()
        df["elbow_score"] = d2.fillna(0).round(4)
        elbow_k = int(df.loc[df["elbow_score"].idxmax(), "k"])
    else:
        elbow_k = int(df["k"].iloc[0])

    best_sil = int(df.loc[df["silhouette"].idxmax(), "k"]) if df["silhouette"].notna().any() \
        else elbow_k

    df.attrs["elbow_k"] = elbow_k
    df.attrs["silhouette_k"] = best_sil
    df.attrs["recommended_k"] = best_sil
    log.info("choose_k: elbow suggests k=%d, silhouette suggests k=%d",
             elbow_k, best_sil)
    return df
