"""Probability distributions implemented from scratch.

scipy is not a dependency of this project, so the CDFs every hypothesis test
needs are implemented here directly. Each is a published numerical recipe with
a stated accuracy bound, and each is validated against textbook values in
tests/test_stats.py rather than against whatever this code happens to return.

The accuracy that matters is in the tails: a p-value of 0.04 versus 0.06 changes
a conclusion, so approximations good only near the centre would be useless.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Normal
# ---------------------------------------------------------------------------


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Density of the normal distribution."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """CDF of the normal distribution, via the error function.

    ``math.erf`` is correctly rounded to double precision in CPython, so this is
    accurate to machine epsilon -- better than any series approximation worth
    writing by hand.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def normal_sf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Survival function, 1 - CDF.

    Computed via erfc rather than as ``1 - cdf`` to avoid catastrophic
    cancellation in the far right tail, where ``1 - 0.9999999999999999`` loses
    every significant digit.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return 0.5 * math.erfc((x - mu) / (sigma * math.sqrt(2.0)))


# Acklam's rational approximation to the inverse normal CDF. Relative error
# below 1.15e-9 across the whole domain, which is far more than any p-value
# needs, and it avoids an iterative root find.
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)

_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW


def normal_ppf(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Inverse normal CDF (quantile function).

    Uses Acklam's approximation followed by one Halley refinement step, which
    brings the result to full double precision.
    """
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError(f"p must lie in (0, 1), got {p}")

    if p < _P_LOW:
        q = math.sqrt(-2 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    elif p <= _P_HIGH:
        q = p - 0.5
        r = q * q
        x = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
            (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)

    # One Halley step against the exact CDF.
    e = normal_cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    x = x - u / (1 + x * u / 2)

    return mu + sigma * x


# ---------------------------------------------------------------------------
# Incomplete beta -- underlies the t and F distributions
# ---------------------------------------------------------------------------


def log_beta(a: float, b: float) -> float:
    """Log of the beta function, via lgamma to avoid overflow."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betacf(a: float, b: float, x: float, *, max_iter: int = 300,
            eps: float = 3e-16) -> float:
    """Continued fraction for the incomplete beta, by the modified Lentz method.

    Numerical Recipes section 6.4. The Lentz formulation is used because the
    naive recurrence underflows: intermediate terms can collapse to zero and
    take the whole fraction with them.
    """
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # Even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        # Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h

    # Non-convergence is worth surfacing rather than returning a silent
    # half-converged value that becomes a wrong p-value.
    raise RuntimeError(f"incomplete beta did not converge for a={a}, b={b}, x={x}")


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"x must lie in [0, 1], got {x}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - log_beta(a, b))
    # The continued fraction converges quickly only for x < (a+1)/(a+b+2);
    # outside that, use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a).
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


# ---------------------------------------------------------------------------
# Student's t
# ---------------------------------------------------------------------------


def t_cdf(t: float, df: float) -> float:
    """CDF of Student's t distribution."""
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    x = df / (df + t * t)
    prob = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - prob if t > 0 else prob


def t_sf(t: float, df: float) -> float:
    return 1.0 - t_cdf(t, df)


def t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value for a t statistic."""
    return 2.0 * (1.0 - t_cdf(abs(t), df))


def t_ppf(p: float, df: float, *, tol: float = 1e-10,
          max_iter: int = 200) -> float:
    """Inverse t CDF, by bisection on the CDF.

    Bisection rather than Newton: it cannot diverge, and at a hundred-odd
    iterations on a monotone function the cost is irrelevant next to being
    certain of convergence.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie in (0, 1), got {p}")
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")

    # The normal quantile is a good starting bracket; widen until it brackets.
    lo, hi = -100.0, 100.0
    guess = normal_ppf(p)
    if math.isfinite(guess):
        lo, hi = guess - 20.0, guess + 20.0

    while t_cdf(lo, df) > p:
        lo -= 50.0
    while t_cdf(hi, df) < p:
        hi += 50.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Incomplete gamma -- underlies chi-squared
# ---------------------------------------------------------------------------


def _gammap_series(a: float, x: float, *, max_iter: int = 500,
                   eps: float = 3e-16) -> float:
    """Series expansion for the lower regularised incomplete gamma P(a, x)."""
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(max_iter):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * eps:
            return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    raise RuntimeError(f"gamma series did not converge for a={a}, x={x}")


def _gammaq_cf(a: float, x: float, *, max_iter: int = 500,
               eps: float = 3e-16) -> float:
    """Continued fraction for the upper regularised incomplete gamma Q(a, x)."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h * math.exp(-x + a * math.log(x) - math.lgamma(a))
    raise RuntimeError(f"gamma continued fraction did not converge for a={a}, x={x}")


def gammainc(a: float, x: float) -> float:
    """Lower regularised incomplete gamma P(a, x).

    The series converges fast for x < a+1 and the continued fraction for larger
    x; using either everywhere is slow or wrong at one end.
    """
    if x < 0 or a <= 0:
        raise ValueError(f"require a > 0 and x >= 0, got a={a}, x={x}")
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        return _gammap_series(a, x)
    return 1.0 - _gammaq_cf(a, x)


# ---------------------------------------------------------------------------
# Chi-squared
# ---------------------------------------------------------------------------


def chi2_cdf(x: float, df: float) -> float:
    """CDF of the chi-squared distribution."""
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    if x <= 0:
        return 0.0
    return gammainc(df / 2.0, x / 2.0)


def chi2_sf(x: float, df: float) -> float:
    """Survival function -- the p-value for a chi-squared statistic."""
    return 1.0 - chi2_cdf(x, df)


def chi2_ppf(p: float, df: float, *, tol: float = 1e-10,
             max_iter: int = 200) -> float:
    """Inverse chi-squared CDF, by bisection."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie in (0, 1), got {p}")
    lo, hi = 0.0, max(10.0 * df, 100.0)
    while chi2_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if chi2_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# F distribution
# ---------------------------------------------------------------------------


def f_cdf(x: float, df1: float, df2: float) -> float:
    """CDF of the F distribution."""
    if x <= 0:
        return 0.0
    if df1 <= 0 or df2 <= 0:
        raise ValueError("degrees of freedom must be positive")
    return betainc(df1 / 2.0, df2 / 2.0, df1 * x / (df1 * x + df2))


def f_sf(x: float, df1: float, df2: float) -> float:
    """Survival function -- the p-value for an ANOVA F statistic."""
    return 1.0 - f_cdf(x, df1, df2)
