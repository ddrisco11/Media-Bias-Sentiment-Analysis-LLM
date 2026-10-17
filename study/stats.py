"""Statistics for the v2 analysis (protocol section 8).

- Near-duplicate clusters: masked, normalised word 3-gram shingles; single linkage at Jaccard >= threshold.
- Cluster bootstrap: resample near-duplicate clusters with replacement within strata (B = 10,000 by default),
  percentile intervals. Estimands are functions of per-stratum means, so standardised (weighted) contrasts work.
- Holm correction, TOST equivalence from bootstrap intervals, DerSimonian-Laird random-effects pooling.
"""
import math
import re

import numpy as np

from study.common import near_duplicate_clusters
from study.mask import mask_outlets

B_DEFAULT = 10_000


def normalise_for_clusters(text):
    text = mask_outlets(text).lower()
    text = re.sub(r'@\w+|\[outlet\]', ' ', text)
    text = re.sub(r'\d+', '0', text)
    return re.sub(r"[^a-z0' ]+", ' ', text)


def assign_clusters(units, group_key, threshold=0.5):
    """units: list of dicts with 'id' and 'text'. Clusters are formed within each group (voice or outlet)."""
    out = {}
    groups = {}
    for u in units:
        groups.setdefault(u[group_key], []).append(u)
    for g, us in groups.items():
        cl = near_duplicate_clusters([u['id'] for u in us], [normalise_for_clusters(u['text']) for u in us], threshold)
        out.update({k: f'{g}:{v}' for k, v in cl.items()})
    return out


class Bootstrap:
    """Cluster bootstrap over a set of units.

    units: list of dicts; value_fn(unit) -> np.array of metric values (NaN = missing for that metric);
    stratum_fn(unit) -> stratum key; cluster key in unit['cluster'].
    estimand(stratum_means: dict[stratum] -> np.array) -> np.array of estimates.
    """

    def __init__(self, units, value_fn, stratum_fn, seed=20241016):
        self.rng = np.random.default_rng(seed)
        self.strata = {}
        for u in units:
            s = stratum_fn(u)
            self.strata.setdefault(s, {}).setdefault(u['cluster'], []).append(value_fn(u))
        # per stratum: arrays of cluster sums and counts (NaN-aware)
        self.packed = {}
        for s, clusters in self.strata.items():
            sums, counts = [], []
            for vals in clusters.values():
                arr = np.array(vals, dtype=float)
                sums.append(np.nansum(arr, axis=0))
                counts.append(np.sum(~np.isnan(arr), axis=0))
            self.packed[s] = (np.array(sums), np.array(counts))

    def means(self, draw=False):
        out = {}
        for s, (sums, counts) in self.packed.items():
            if draw:
                idx = self.rng.integers(0, len(sums), len(sums))
                tot, n = sums[idx].sum(0), counts[idx].sum(0)
            else:
                tot, n = sums.sum(0), counts.sum(0)
            with np.errstate(invalid='ignore', divide='ignore'):
                out[s] = tot / n
        return out

    def run(self, estimand, B=B_DEFAULT, alpha=0.05):
        point = np.asarray(estimand(self.means()), dtype=float)
        draws = np.array([estimand(self.means(draw=True)) for _ in range(B)], dtype=float)
        lo = np.nanpercentile(draws, 100 * alpha / 2, axis=0)
        hi = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0)
        lo90 = np.nanpercentile(draws, 5, axis=0)
        hi90 = np.nanpercentile(draws, 95, axis=0)
        # two-sided bootstrap p-value for H0: estimate = 0
        p = 2 * np.minimum(np.nanmean(draws <= 0, axis=0), np.nanmean(draws >= 0, axis=0))
        se = np.nanstd(draws, axis=0, ddof=1)
        return {'estimate': point, 'ci95': np.stack([lo, hi], -1), 'ci90': np.stack([lo90, hi90], -1),
                'p_boot': np.minimum(1.0, p), 'se': se}


def holm(pvalues):
    """Holm-adjusted p-values (same order as input)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def tost_equivalent(ci90, margin):
    """Equivalent if the 90% interval lies entirely within [-margin, margin]."""
    lo, hi = ci90
    return bool(lo > -margin and hi < margin)


def random_effects(estimates, ses):
    """DerSimonian-Laird pooled estimate, 95% CI and tau^2."""
    y = np.asarray(estimates, float)
    # Cells where the judge rated every tweet 0 give a bootstrap SE of exactly 0; floor it so pooling is defined.
    s = np.maximum(np.asarray(ses, float), 1e-6)
    w = 1 / s ** 2
    fixed = np.sum(w * y) / np.sum(w)
    q = np.sum(w * (y - fixed) ** 2)
    k = len(y)
    tau2 = max(0.0, (q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w))) if k > 1 else 0.0
    wr = 1 / (s ** 2 + tau2)
    pooled = np.sum(wr * y) / np.sum(wr)
    se = math.sqrt(1 / np.sum(wr))
    return {'estimate': float(pooled), 'ci95': [float(pooled - 1.96 * se), float(pooled + 1.96 * se)], 'se': float(se),
            'tau2': float(tau2), 'k': k}


def js_divergence(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = (p + q) / 2
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def total_variation(p, q):
    return 0.5 * float(np.sum(np.abs(np.asarray(p, float) - np.asarray(q, float))))
