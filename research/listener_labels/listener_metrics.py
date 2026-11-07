"""Metrics that compare a model's probabilities with audio-only listener votes (see protocol-zh.txt)."""
import numpy as np
from scipy.stats import spearmanr

SET_THRESHOLD = 0.2
GROUPS = {'low': (0.0, 0.5), 'mid': (0.5, 0.7), 'high': (0.7, 1.01)}


def as_sets(dist):
    """Classes with share >= 0.2, always including the top class (ties kept)."""
    return (dist >= SET_THRESHOLD) | (dist == dist.max(1, keepdims=True))


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def per_clip(p, y, q):
    """Per-clip quantities; averaged later over clips, actors or consensus groups."""
    pred = p.argmax(1)
    rows = np.arange(len(y))
    plural = q == q.max(1, keepdims=True)
    S, T = as_sets(p), as_sets(q)
    inter, union = (S & T).sum(1), (S | T).sum(1)
    return {'actor_correct': (pred == y).astype(float),
            'in_listener_plurality': plural[rows, pred].astype(float),
            'ce_to_votes': -(q * np.log(np.clip(p, 1e-9, 1))).sum(1),
            'brier_to_votes': ((p - q) ** 2).sum(1),
            'jaccard': inter / union,
            'set_recall': inter / T.sum(1),
            'set_size': S.sum(1).astype(float),
            'max_prob': p.max(1)}


def evaluate(p, y, q, speakers):
    c = per_clip(p, y, q)
    share = q.max(1)
    out = {k: float(v.mean()) for k, v in c.items()}
    out['ece_vs_actor'] = ece(p, y)
    out['awareness_rho_maxprob_vs_listener_share'] = float(spearmanr(c['max_prob'], share).statistic)
    out['groups'] = {}
    for g, (lo, hi) in GROUPS.items():
        m = (share >= lo) & (share < hi)
        out['groups'][g] = {'n': int(m.sum()), **{k: float(v[m].mean()) for k, v in c.items()}}
    actors = sorted(set(speakers.tolist()))
    out['per_actor'] = {a: {k: float(v[speakers == a].mean()) for k, v in c.items()} for a in actors}
    return out


def actor_bootstrap(diff, seed=20261111, reps=5000):
    """diff: per-actor differences (already averaged over seeds)."""
    diff = np.asarray(diff, float)
    draws = np.random.default_rng(seed).integers(0, len(diff), (reps, len(diff)))
    lo, hi = np.quantile(diff[draws].mean(1), [.025, .975])
    return {'mean': float(diff.mean()), 'ci95': [float(lo), float(hi)], 'n_actors': len(diff)}
