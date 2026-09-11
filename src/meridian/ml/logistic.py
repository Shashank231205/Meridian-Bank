"""Logistic regression by IRLS, with inference.

Fitted by iteratively reweighted least squares -- Newton-Raphson on the
log-likelihood -- which is the standard method and converges in a handful of
iterations for well-conditioned problems.

The reason to write this rather than import scikit-learn is inference.
``LogisticRegression`` gives coefficients and no standard errors, so it cannot
answer "is this driver statistically distinguishable from zero", which is the
question a decision-management team actually asks. IRLS produces the Hessian as
a by-product of fitting; inverting it gives the covariance matrix, and from that
standard errors, Wald z statistics, p-values and confidence intervals on the
odds ratios. That is the difference between a model that predicts and a model
you can reason about.

Numpy only, no scipy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..common.logging import get_logger
from ..stats.distributions import chi2_sf, normal_ppf, normal_sf

log = get_logger(__name__)


@dataclass
class LogisticFit:
    """A fitted logistic regression with full inference."""

    feature_names: list[str]
    coefficients: np.ndarray          # includes the intercept at position 0
    standard_errors: np.ndarray
    z_values: np.ndarray
    p_values: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    n_obs: int
    n_features: int
    log_likelihood: float
    null_log_likelihood: float
    n_iterations: int
    converged: bool
    separation_warning: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    # --- goodness of fit ---------------------------------------------------

    @property
    def deviance(self) -> float:
        return -2.0 * self.log_likelihood

    @property
    def null_deviance(self) -> float:
        return -2.0 * self.null_log_likelihood

    @property
    def mcfadden_r2(self) -> float:
        """McFadden's pseudo-R2.

        Not comparable to an OLS R2 -- values of 0.2-0.4 indicate an excellent
        fit here, and reading 0.15 as "explains 15% of variance" is wrong.
        """
        if self.null_log_likelihood == 0:
            return 0.0
        return 1.0 - self.log_likelihood / self.null_log_likelihood

    @property
    def aic(self) -> float:
        return -2.0 * self.log_likelihood + 2 * len(self.coefficients)

    @property
    def bic(self) -> float:
        return (-2.0 * self.log_likelihood
                + len(self.coefficients) * np.log(self.n_obs))

    @property
    def lr_statistic(self) -> float:
        """Likelihood-ratio statistic against the intercept-only model."""
        return 2.0 * (self.log_likelihood - self.null_log_likelihood)

    @property
    def lr_p_value(self) -> float:
        return chi2_sf(self.lr_statistic, self.n_features)

    # --- presentation ------------------------------------------------------

    @property
    def odds_ratios(self) -> np.ndarray:
        return np.exp(self.coefficients)

    def summary_frame(self) -> pd.DataFrame:
        """The coefficient table, as a statistician would expect to read it."""
        return pd.DataFrame({
            "term": self.feature_names,
            "coefficient": np.round(self.coefficients, 6),
            "std_error": np.round(self.standard_errors, 6),
            "z_value": np.round(self.z_values, 4),
            "p_value": self.p_values,
            "odds_ratio": np.round(np.exp(self.coefficients), 6),
            "or_ci_low": np.round(np.exp(self.ci_low), 6),
            "or_ci_high": np.round(np.exp(self.ci_high), 6),
            "significant": self.p_values < 0.05,
        })

    def summary(self) -> str:
        lines = [
            "Logistic regression (IRLS)",
            f"  observations      {self.n_obs:,}",
            f"  features          {self.n_features}",
            f"  converged         {self.converged} in {self.n_iterations} iterations",
            f"  log-likelihood    {self.log_likelihood:.4f}",
            f"  null log-lik      {self.null_log_likelihood:.4f}",
            f"  McFadden R2       {self.mcfadden_r2:.4f}",
            f"  AIC / BIC         {self.aic:.2f} / {self.bic:.2f}",
            f"  LR test           chi2={self.lr_statistic:.2f}, "
            f"df={self.n_features}, p={self.lr_p_value:.4g}",
            "",
        ]
        if self.separation_warning:
            lines.append(
                "  WARNING: near-perfect separation detected. Coefficients and "
                "standard errors for the affected terms are not trustworthy."
            )
            lines.append("")
        lines.append(self.summary_frame().to_string(index=False))
        return "\n".join(lines)

    # --- prediction --------------------------------------------------------

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predicted probabilities. X must not include an intercept column."""
        X = _as_array(X)
        design = np.column_stack([np.ones(len(X)), X])
        return _sigmoid(design @ self.coefficients)

    def predict(self, X, *, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def fit_logistic(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    feature_names: list[str] | None = None,
    max_iter: int = 50,
    tol: float = 1e-8,
    ridge: float = 1e-8,
    alpha: float = 0.05,
) -> LogisticFit:
    """Fit logistic regression by iteratively reweighted least squares.

    Args:
        X: design matrix WITHOUT an intercept column; one is prepended.
        y: binary outcome, 0/1.
        ridge: tiny ridge penalty on the Hessian for numerical stability. Left
            at 1e-8 it does not shift estimates meaningfully but keeps the
            solve from failing on a near-singular Hessian, which happens
            whenever two features are nearly collinear.
    """
    X_arr = _as_array(X)
    y_arr = np.asarray(pd.Series(y).to_numpy(), dtype=float).ravel()

    if len(X_arr) != len(y_arr):
        raise ValueError(f"X has {len(X_arr)} rows, y has {len(y_arr)}")
    if not np.isin(np.unique(y_arr), [0, 1]).all():
        raise ValueError("y must be binary 0/1")
    if len(np.unique(y_arr)) < 2:
        raise ValueError("y is constant: nothing to fit")

    names = list(feature_names) if feature_names is not None else (
        list(X.columns) if isinstance(X, pd.DataFrame)
        else [f"x{i + 1}" for i in range(X_arr.shape[1])]
    )
    names = ["(intercept)"] + names

    n, k = X_arr.shape
    design = np.column_stack([np.ones(n), X_arr])
    beta = np.zeros(k + 1)

    converged = False
    iterations = 0

    for iterations in range(1, max_iter + 1):
        eta = design @ beta
        mu = _sigmoid(eta)

        # IRLS weights: w = mu(1-mu). Clipped away from zero because a fitted
        # probability of exactly 0 or 1 gives a zero weight, a singular Hessian
        # and a crash instead of a diagnosable separation warning.
        w = np.clip(mu * (1 - mu), 1e-10, None)

        # Working response for the weighted least-squares step.
        z = eta + (y_arr - mu) / w

        XtW = design.T * w
        hessian = XtW @ design + ridge * np.eye(k + 1)
        rhs = XtW @ z

        try:
            beta_new = np.linalg.solve(hessian, rhs)
        except np.linalg.LinAlgError:
            # Fall back to the pseudo-inverse: a singular Hessian means
            # collinearity, and lstsq gives the minimum-norm answer rather than
            # failing outright.
            beta_new = np.linalg.lstsq(hessian, rhs, rcond=None)[0]
            log.warning("singular Hessian at iteration %d; used lstsq", iterations)

        delta = np.max(np.abs(beta_new - beta))
        beta = beta_new
        if delta < tol:
            converged = True
            break

    if not converged:
        log.warning("IRLS did not converge in %d iterations (delta=%.2e)",
                    max_iter, delta)

    # --- inference ---------------------------------------------------------
    eta = design @ beta
    mu = _sigmoid(eta)
    w = np.clip(mu * (1 - mu), 1e-10, None)

    hessian = (design.T * w) @ design + ridge * np.eye(k + 1)
    try:
        # The covariance matrix is the inverse of the Fisher information, which
        # is exactly the Hessian IRLS already computed. This is the step
        # sklearn omits and the reason this module exists.
        cov = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(hessian)
        log.warning("Hessian not invertible; standard errors use the pseudo-inverse")

    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        z_values = np.where(se > 0, beta / se, 0.0)
    p_values = np.array([2.0 * normal_sf(abs(zi)) for zi in z_values])

    crit = normal_ppf(1 - alpha / 2)
    ci_low, ci_high = beta - crit * se, beta + crit * se

    # Separation check: enormous coefficients with enormous standard errors are
    # the signature of a feature that perfectly predicts the outcome, and the
    # resulting numbers look impressive while meaning nothing.
    separation = bool(np.any(np.abs(beta[1:]) > 15) or np.any(se[1:] > 50))
    if separation:
        log.warning("possible separation: some coefficients or SEs are extreme")

    ll = _log_likelihood(y_arr, mu)
    p_bar = y_arr.mean()
    null_ll = float(np.sum(y_arr * np.log(p_bar) + (1 - y_arr) * np.log(1 - p_bar)))

    log.info(
        "logistic fit: n=%s, k=%d, converged=%s in %d iters, McFadden R2=%.4f",
        f"{n:,}", k, converged, iterations,
        1 - ll / null_ll if null_ll else 0.0,
    )

    return LogisticFit(
        feature_names=names, coefficients=beta, standard_errors=se,
        z_values=z_values, p_values=p_values, ci_low=ci_low, ci_high=ci_high,
        n_obs=n, n_features=k, log_likelihood=ll, null_log_likelihood=null_ll,
        n_iterations=iterations, converged=converged,
        separation_warning=separation,
        extra={"event_rate": float(p_bar)},
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function.

    exp(-x) overflows for x below about -745. Splitting on the sign keeps every
    exponent negative, which cannot overflow.
    """
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _log_likelihood(y: np.ndarray, mu: np.ndarray) -> float:
    eps = 1e-12
    mu = np.clip(mu, eps, 1 - eps)
    return float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))


def _as_array(X) -> np.ndarray:
    arr = X.to_numpy(dtype=float) if isinstance(X, (pd.DataFrame, pd.Series)) \
        else np.asarray(X, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def standardise(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Centre and scale features, returning the means and scales used.

    Standardising makes coefficients comparable in magnitude -- which is what
    "the strongest driver" means -- and improves IRLS conditioning when features
    differ by orders of magnitude, as income in rupees and a 0-1 engagement
    score do.
    """
    mean = X.mean()
    sd = X.std(ddof=0).replace(0, 1.0)
    return (X - mean) / sd, mean, sd
