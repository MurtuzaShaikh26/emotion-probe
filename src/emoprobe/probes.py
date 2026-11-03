"""Probe maths: mean-difference directions, logistic probes, AUROC, cluster bootstrap, subspace projection."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
C_GRID = (1e-3, 1e-2, 1e-1, 1.0)
N_BOOT = 1000


# ---------------------------------------------------------------- directions
def mean_diff(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Emotional (1) minus neutral (0) class mean."""
    return X[y == 1].mean(0) - X[y == 0].mean(0)


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(unit(a) @ unit(b))


class Direction:
    """Classify by projecting onto a direction; threshold = midpoint of the training class means."""

    def __init__(self, v: np.ndarray, Xtr: np.ndarray, ytr: np.ndarray):
        self.v = unit(v)
        s = Xtr @ self.v
        self.thr = 0.5 * (s[ytr == 1].mean() + s[ytr == 0].mean())

    def score(self, X):
        return X @ self.v

    def predict(self, X):
        return (self.score(X) > self.thr).astype(int)


class Probe:
    """Standardise + L2 logistic regression. C picked by grouped CV unless given."""

    def __init__(self, Xtr, ytr, groups=None, C: float | None = None, multiclass=False):
        self.sc = StandardScaler().fit(Xtr)
        Z = self.sc.transform(Xtr)
        if C is None:
            C = pick_C(Z, ytr, groups)
        self.C = C
        self.lr = LogisticRegression(C=C, max_iter=3000).fit(Z, ytr)

    def score(self, X):
        return self.lr.decision_function(self.sc.transform(X))

    def predict(self, X):
        return self.lr.predict(self.sc.transform(X))


def pick_C(Z, y, groups, n_splits=5) -> float:
    gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    best, bestC = -1.0, C_GRID[0]
    for C in C_GRID:  # ascending: ties keep the stronger regularisation
        accs = []
        for tr, te in gkf.split(Z, y, groups):
            m = LogisticRegression(C=C, max_iter=3000).fit(Z[tr], y[tr])
            accs.append((m.predict(Z[te]) == y[te]).mean())
        if np.mean(accs) > best + 1e-9:
            best, bestC = np.mean(accs), C
    return bestC


# ---------------------------------------------------------------- metrics
def auroc_batch(S: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Rank-based AUROC along the last axis. S, Y: [B, n]."""
    R = rankdata(S, axis=-1)
    n1 = Y.sum(-1)
    n0 = Y.shape[-1] - n1
    return ((R * Y).sum(-1) - n1 * (n1 + 1) / 2) / np.maximum(n1 * n0, 1)


def auroc(s, y) -> float:
    return float(auroc_batch(np.asarray(s)[None], np.asarray(y)[None])[0])


def boot_idx(items: np.ndarray, n_boot=N_BOOT, seed=0) -> np.ndarray:
    """Cluster bootstrap: resample ITEMS (a task or D2 pair, both labels together). Returns [B, n] example idx."""
    uniq, inv = np.unique(items, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    sizes = {len(m) for m in members}
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    if len(sizes) == 1:
        M = np.stack(members)  # [n_items, size]
        return M[pick].reshape(n_boot, -1)
    return np.stack([np.concatenate([members[k] for k in row]) for row in pick])


def summarize(scores, y, items, thr=None, pred=None, seed=0) -> dict:
    """AUROC (threshold-free) and accuracy (fixed threshold / predictions) with 95% bootstrap CIs."""
    scores, y = np.asarray(scores, float), np.asarray(y, int)
    B = boot_idx(np.asarray(items), seed=seed)
    out = {}
    a = auroc_batch(scores[B], y[B])
    out["auroc"] = [auroc(scores, y), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    if pred is None and thr is not None:
        pred = (scores > thr).astype(int)
    if pred is not None:
        correct = (np.asarray(pred) == y).astype(float)
        acc = correct[B].mean(1)
        out["acc"] = [float(correct.mean()), float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))]
    return out


def acc_ci(correct, items, seed=0) -> list:
    correct = np.asarray(correct, float)
    B = boot_idx(np.asarray(items), seed=seed)
    acc = correct[B].mean(1)
    return [float(correct.mean()), float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))]


# ---------------------------------------------------------------- subspace
def orth_basis(V: np.ndarray, rel_tol=1e-6) -> np.ndarray:
    """Rows of V (unit-normalised first) -> orthonormal basis rows, ordered by singular value."""
    Vn = np.stack([unit(v) for v in V])
    U, S, Wt = np.linalg.svd(Vn, full_matrices=False)
    keep = S > rel_tol * S[0]
    return Wt[keep], S[keep]


def project_out(v: np.ndarray, B: np.ndarray) -> np.ndarray:
    return v - B.T @ (B @ v)
