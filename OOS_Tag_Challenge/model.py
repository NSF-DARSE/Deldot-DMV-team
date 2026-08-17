"""Ordinal (proportional-odds) classifier for the three review classes.

The three classes sit on one axis: the strength of evidence that the person is
now operating in Delaware.

    review_not_warranted (0)  <  insufficient_evidence (1)  <  review_warranted (2)

The development labels show that cases never jump from `review_not_warranted`
straight to `review_warranted` between phases -- they always pass through
`insufficient_evidence`. A cumulative-link model encodes exactly that structure
and needs far fewer labelled rows than an unordered 3-way classifier.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

CLASSES = ["review_not_warranted", "insufficient_evidence", "review_warranted"]
CLASS_TO_ORD = {c: i for i, c in enumerate(CLASSES)}


class OrdinalLogit:
    """Proportional-odds model with an L2 penalty on the slopes.

    P(y <= k) = sigmoid(theta_k - x . beta), theta_0 < theta_1.
    """

    def __init__(self, C=1.0, max_iter=500):
        self.C = C
        self.max_iter = max_iter

    def _unpack(self, p, d):
        beta = p[:d]
        t0 = p[d]
        t1 = t0 + np.exp(p[d + 1])          # enforces theta_0 < theta_1
        return beta, t0, t1

    def _nll(self, p, X, y, sw):
        d = X.shape[1]
        beta, t0, t1 = self._unpack(p, d)
        z = X @ beta
        c0 = expit(t0 - z)
        c1 = expit(t1 - z)
        pr = np.empty((len(y), 3))
        pr[:, 0] = c0
        pr[:, 1] = c1 - c0
        pr[:, 2] = 1.0 - c1
        pr = np.clip(pr, 1e-9, 1.0)
        ll = -np.sum(sw * np.log(pr[np.arange(len(y)), y]))
        return ll + 0.5 * np.dot(beta, beta) / self.C

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        d = X.shape[1]
        p0 = np.zeros(d + 2)
        p0[d] = -0.4
        p0[d + 1] = 0.0
        r = minimize(self._nll, p0, args=(X, y, sw), method="L-BFGS-B",
                     options={"maxiter": self.max_iter})
        self.coef_, self.theta0_, self.theta1_ = self._unpack(r.x, d)
        self.result_ = r
        return self

    def decision_function(self, X):
        return np.asarray(X, float) @ self.coef_

    def predict_proba(self, X):
        z = self.decision_function(X)
        c0 = expit(self.theta0_ - z)
        c1 = expit(self.theta1_ - z)
        return np.clip(np.column_stack([c0, c1 - c0, 1.0 - c1]), 1e-9, 1.0)
