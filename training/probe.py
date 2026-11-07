"""Pilot study: does the speech-emotion model use answer-relevant evidence?

Adapts the diagnostics of ReaLVR (Xiao et al., 2026, arXiv:2609.34563) from latent
visual tokens to the K=8 latent segment slots of SegmentMLP. Everything runs on the
held-out TEST speakers only.

 1. Paired counterfactual behaviour  (paper App. J.1, Table J.1)
    C pairs: same speaker + same sentence, different emotion  -> answer SHOULD change
    U pairs: same speaker + same emotion, different sentence  -> answer should NOT change
    Reports accuracy, prediction change, strict paired correct flip, false flip,
    and mean latent cosine distance between the two members (Table G.5).
 2. Fixed-context slot replacement   (App. G.1 / Fig. I.3b)
    Replace the top-k slots (ranked by answer contrast, raw read-out, or at random) with
    the training-mean slot at that position; other slots fixed. Report the drop in
    correct-answer log-probability.
 3. Slot exchange                    (App. J.2 / Table J.2)
    For C pairs, copy k donor slots into the recipient (same positions) and measure the
    lift in the frequency of the donor's answer. Controls: sham self-swap (zero by
    construction for this deterministic model) and an unrelated donor.
 4. Evidence ablations / nuisance perturbations
    Neutralise F0 or energy features, flatten the F0 contour across segments, or add
    answer-preserving noise / gain changes to the waveform.

    python3 training/probe.py      # writes model/probe_results.json and model/probe_*.png
"""
import json
import multiprocessing as mp
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

try:
    from . import features as F
    from .inference import Predictor
    from .train import load_data, speaker_split
except ImportError:
    import features as F
    from inference import Predictor
    from train import load_data, speaker_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'model'
KS = [0, 1, 2, 4, 8]
F0_IDX = [F.FEATURE_NAMES.index(n) for n in ('logf0_mean', 'logf0_std', 'logf0_range', 'voiced_frac')]
EN_IDX = [F.FEATURE_NAMES.index(n) for n in ('logrms_mean', 'logrms_std')]


# ----------------------------------------------------------------------------- pairs
def build_pairs(meta, test_idx):
    """C pairs differ only in emotion; U pairs differ only in sentence. The IEO sentence
    (recorded at several intensities) is excluded so every clip has level 'XX'."""
    keep = [i for i in test_idx if meta['sentence'][i] != 'IEO']
    by_ss, by_se = {}, {}
    for i in keep:
        by_ss.setdefault((meta['speaker'][i], meta['sentence'][i]), []).append(i)
        by_se.setdefault((meta['speaker'][i], meta['emotion'][i]), []).append(i)
    C = [p for g in by_ss.values() for p in combinations(sorted(g), 2)]
    U = [p for g in by_se.values() for p in combinations(sorted(g), 2)]
    return np.array(C), np.array(U)


def paired_counterfactual(pred, y, Z, C, U):
    def cosdist(a, b):
        a, b = Z[a].reshape(len(a), -1), Z[b].reshape(len(b), -1)
        return float(np.mean(1 - (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))))
    a, b = C[:, 0], C[:, 1]
    ua, ub = U[:, 0], U[:, 1]
    return {
        'n_C': len(C), 'n_U': len(U),
        'acc_on_C_clips': float(np.mean(np.r_[pred[a] == y[a], pred[b] == y[b]])),
        'prediction_change_C': float(np.mean(pred[a] != pred[b])),
        'strict_correct_flip_C': float(np.mean((pred[a] == y[a]) & (pred[b] == y[b]))),
        'false_flip_U': float(np.mean(pred[ua] != pred[ub])),
        'latent_distance_C': cosdist(a, b),
        'latent_distance_U': cosdist(ua, ub),
    }


# ----------------------------------------------------------------------------- slot interventions
def replacement_curves(P, Z, y, zbar, n_random=20, seed=0):
    """Correct-answer log-prob drop after replacing top-k slots (others fixed)."""
    rng = np.random.default_rng(seed)
    z = torch.as_tensor(Z)
    logp0, r = P.from_latents(z)
    r = r.numpy()
    lp0 = logp0.numpy()[np.arange(len(y)), y]
    others = np.array([[c for c in range(r.shape[2]) if c != yy] for yy in y])
    r_true = r[np.arange(len(y)), :, y]                                           # (N, K)
    r_other = np.take_along_axis(r, others[:, None, :].repeat(r.shape[1], 1), 2).mean(2)
    rankings = {'answer_contrast': np.argsort(-np.maximum(r_true - r_other, 0) - 1e-9 * r_true, 1),
                'raw_readout': np.argsort(-r_true, 1)}
    out = {}

    def drop(order, k):
        zz = z.clone()
        if k:
            sel = torch.as_tensor(order[:, :k])
            rows = torch.arange(len(y))[:, None].expand_as(sel)
            zz[rows, sel] = torch.as_tensor(zbar)[sel]
        lp = P.from_latents(zz)[0].numpy()[np.arange(len(y)), y]
        return float(np.mean(lp0 - lp)), float(np.mean(np.exp(lp)))

    for name, order in rankings.items():
        out[name] = {k: drop(order, k) for k in KS}
    rand = {k: [] for k in KS}
    for _ in range(n_random):
        order = np.argsort(rng.random(r_true.shape), 1)
        for k in KS:
            rand[k].append(drop(order, k))
    out['random'] = {k: tuple(np.mean(v, 0).tolist()) for k, v in rand.items()}
    return out


def exchange(P, Z, y, C, meta, idx_pool, n_random=10, seed=0):
    """Copy donor slots into the recipient at the same positions (both directions of every C pair)."""
    rng = np.random.default_rng(seed)
    rec = np.r_[C[:, 0], C[:, 1]]
    don = np.r_[C[:, 1], C[:, 0]]
    z = torch.as_tensor(Z)
    logp, r = P.from_latents(z[rec])
    r = r.numpy()
    yr, yd = y[rec], y[don]
    # positions most read for the recipient's own answer relative to the donor's answer
    contrast = np.maximum(r[np.arange(len(rec)), :, yr] - r[np.arange(len(rec)), :, yd], 0)

    def run(order, k, donor_idx):
        zz = z[rec].clone()
        sel = torch.as_tensor(order[:, :k])
        rows = torch.arange(len(rec))[:, None].expand_as(sel)
        zz[rows, sel] = z[donor_idx][rows, sel]
        lp = P.from_latents(zz)[0].numpy()
        return lp

    base = logp.numpy()
    base_hit = base.argmax(1) == yd
    base_margin = base[np.arange(len(rec)), yd] - base[np.arange(len(rec)), yr]

    def summarise(lp):
        hit = lp.argmax(1) == yd
        margin = lp[np.arange(len(rec)), yd] - lp[np.arange(len(rec)), yr]
        return {'donor_answer_lift_pp': float(100 * (hit.mean() - base_hit.mean())),
                'donor_margin_shift_nats': float(np.mean(margin - base_margin))}

    res = {'n_swaps': int(len(rec)), 'sham_self_swap': {'donor_answer_lift_pp': 0.0, 'donor_margin_shift_nats': 0.0}}
    for k in (1, 2, 4):
        res[f'contrast_k{k}'] = summarise(run(np.argsort(-contrast, 1), k, don))
        rnd = [summarise(run(np.argsort(rng.random(contrast.shape), 1), k, don)) for _ in range(n_random)]
        res[f'random_k{k}'] = {m: float(np.mean([x[m] for x in rnd])) for m in rnd[0]}
    res['full_span_k8'] = summarise(run(np.tile(np.arange(8), (len(rec), 1)), 8, don))
    # unrelated donor: different speaker, different sentence, emotion != recipient and != donor
    unrelated = []
    for i, j in zip(rec, don):
        cand = [c for c in idx_pool if meta['speaker'][c] != meta['speaker'][i]
                and meta['sentence'][c] != meta['sentence'][i] and y[c] not in (y[i], y[j])]
        unrelated.append(rng.choice(cand))
    res['unrelated_donor_k8'] = summarise(run(np.tile(np.arange(8), (len(rec), 1)), 8, np.array(unrelated)))
    return res


# ----------------------------------------------------------------------------- perturbations
def _noise_feats(args):
    path, kind = args
    y = F.load(path)
    if kind == 'noise_20dB':
        rng = np.random.default_rng(abs(hash(Path(path).name)) % 2**32)
        p = np.mean(y ** 2)
        y = y + rng.normal(0, np.sqrt(p / 10 ** (20 / 10)), len(y)).astype(np.float32)
    elif kind == 'gain_minus6dB':
        y = y * 10 ** (-6 / 20)
    return F.extract(y)


def perturbations(Ps, X, y, files, workers=2):
    """Prediction change under evidence ablations (feature level) and nuisance changes (waveform)."""
    def neutralise(Xp, cols, mode):
        Xp = Xp.copy()
        if mode == 'flatten':      # keep utterance-level value, remove change across segments
            Xp[:, :, cols] = Xp[:, :, cols].mean(1, keepdims=True)
        return Xp

    variants = {}
    for name, cols, mode in [('flatten_f0_contour', F0_IDX[:1], 'flatten'),
                             ('flatten_energy_contour', EN_IDX[:1], 'flatten')]:
        variants[name] = neutralise(X, cols, mode)
    paths = [str(ROOT / 'data' / 'crema' / f) for f in files]
    with mp.Pool(workers) as pool:
        for kind in ('noise_20dB', 'gain_minus6dB'):
            variants[kind] = np.stack(pool.map(_noise_feats, [(p, kind) for p in paths], chunksize=32))
    results = {}
    for mname, P in Ps.items():
        p0 = P.predict_features(X)[0]
        # also: remove pitch/energy *levels* by setting them to the training mean (standardised 0)
        res = {}
        for vname, Xv in variants.items():
            res[vname] = _compare(p0, P.predict_features(Xv)[0], y)
        for vname, cols in [('neutralise_f0', F0_IDX), ('neutralise_energy', EN_IDX)]:
            Xs = P.standardise(X)
            used = [P.fidx.index(c) for c in cols if c in P.fidx]
            Xs[:, :, used] = 0.0
            z = P.latents(Xs)
            res[vname] = _compare(p0, P.from_latents(z)[0].exp().numpy(), y)
        results[mname] = res
    return results


def _compare(p0, p1, y):
    m = 0.5 * (p0 + p1)
    js = 0.5 * (p0 * np.log(p0 / m)).sum(1) + 0.5 * (p1 * np.log(p1 / m)).sum(1)
    return {'acc_before': float(np.mean(p0.argmax(1) == y)), 'acc_after': float(np.mean(p1.argmax(1) == y)),
            'prediction_change': float(np.mean(p0.argmax(1) != p1.argmax(1))), 'mean_js': float(js.mean())}


# ----------------------------------------------------------------------------- figures
def plot(results, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    rep = results['replacement']['all']
    for name, style in [('answer_contrast', '-o'), ('raw_readout', '-s'), ('random', '--^')]:
        ax[0].plot(KS, [rep[name][k][0] for k in KS], style, label=name.replace('_', ' '))
    ax[0].set(title='Replace top-k segment slots (others fixed)', xlabel='k replaced slots',
              ylabel='correct-answer log-prob drop (nats)')
    ex = results['exchange']['all']
    xs = [1, 2, 4]
    ax[1].plot(xs, [ex[f'contrast_k{k}']['donor_answer_lift_pp'] for k in xs], '-o', label='answer contrast')
    ax[1].plot(xs, [ex[f'random_k{k}']['donor_answer_lift_pp'] for k in xs], '--^', label='random')
    ax[1].axhline(ex['full_span_k8']['donor_answer_lift_pp'], color='gray', ls=':', label='full span (k=8)')
    ax[1].axhline(ex['unrelated_donor_k8']['donor_answer_lift_pp'], color='firebrick', ls=':', label='unrelated donor')
    ax[1].set(title='Swap slots from the other delivery', xlabel='k swapped slots', ylabel='donor-answer lift (pp)')
    for a in ax:
        a.legend(frameon=False, fontsize=8)
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main():
    X, y, meta = load_data()
    groups = speaker_split(meta['speaker'])
    test = np.where(np.isin(meta['speaker'], groups['test']))[0]
    train = np.where(np.isin(meta['speaker'], groups['train']))[0]
    Ps = {'all': Predictor(OUT / 'segment_mlp.pt')}
    if (OUT / 'mfcc_only' / 'segment_mlp.pt').exists():
        Ps['mfcc_only'] = Predictor(OUT / 'mfcc_only' / 'segment_mlp.pt')
    C, U = build_pairs(meta, test)
    yt = y[test]
    pos = {g: i for i, g in enumerate(test)}
    Cl = np.vectorize(pos.get)(C)
    Ul = np.vectorize(pos.get)(U)
    results = {'paired': {}, 'replacement': {}, 'exchange': {}}
    for name, P in Ps.items():
        probs, _, Z = P.predict_features(X[test])
        pred = probs.argmax(1)
        results['paired'][name] = paired_counterfactual(pred, yt, Z, Cl, Ul)
        zbar = P.latents(P.standardise(X[train])).numpy().mean(0)
        results['replacement'][name] = replacement_curves(P, Z, yt, zbar)
        meta_t = {k: v[test] for k, v in meta.items()}
        results['exchange'][name] = exchange(P, Z, yt, Cl, meta_t, np.arange(len(test)))
        print(name, json.dumps(results['paired'][name], indent=1))
    print('perturbations (waveform re-extraction)...', flush=True)
    results['perturbation'] = perturbations(Ps, X[test], yt, meta['fname'][test])
    (OUT / 'probe_results.json').write_text(json.dumps(results, indent=1, default=float))
    plot(results, OUT / 'probe_interventions.png')
    print(json.dumps({k: results[k] for k in ('replacement', 'exchange', 'perturbation')}, indent=1, default=float))


if __name__ == '__main__':
    main()
