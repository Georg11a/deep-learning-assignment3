"""HW3 analysis: robustness, attention diagnostics, 2x2 path swap and listener-label failure taxonomy.

All analyses use the held-out TEST speakers only. Requires:
  hw3/runs/*/model.pt          (python3 hw3/train_variants.py)
  data/hw3_perturbed.npz       (python3 hw3/perturb_features.py)
  data/tabulatedVotes.csv      (CREMA-D processedResults; audio-only rows have ids starting with '1')

    python3 hw3/analysis.py     # writes hw3/results/analysis.json
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
sys.path.insert(0, str(ROOT / 'hw3'))
import features as F  # noqa: E402
from train import load_data, speaker_split  # noqa: E402
from probe import build_pairs  # noqa: E402
from train_variants import load_model, RUNS  # noqa: E402
from models import POOLINGS  # noqa: E402

OUT = ROOT / 'hw3' / 'results'
SEEDS = (7, 11, 23)
VOTE_CODES = ['A', 'D', 'F', 'H', 'N', 'S']          # same order as F.EMOTIONS
LOGRMS = F.FEATURE_NAMES.index('logrms_mean')
VOICED = F.FEATURE_NAMES.index('voiced_frac')
LOGF0 = F.FEATURE_NAMES.index('logf0_mean')


# ----------------------------------------------------------------------------- helpers
def audio_votes(fnames):
    rows = {r['fileName']: r for r in csv.DictReader(open(ROOT / 'data' / 'tabulatedVotes.csv'))
            if int(r['']) // 100000 == 1}
    return np.array([[int(rows[Path(f).stem][c]) for c in VOTE_CODES] for f in fnames])


@torch.no_grad()
def parts(model, Xs):
    """logits, z (N,K,H), alpha (N,K) or None, per-segment contributions (N,K,C)."""
    x = torch.as_tensor(Xs, dtype=torch.float32)
    logits, z, a = model(x, return_parts=True)
    W = model.head.weight
    if model.pooling == 'concat':
        contrib = torch.einsum('bkh,ckh->bkc', z, W.view(W.shape[0], model.k, -1))
    else:
        contrib = a.unsqueeze(-1) * torch.einsum('bkh,ch->bkc', z, W)
    return logits.numpy(), z, a, contrib.numpy()


def margin(logits, y):
    """true-class logit minus the best other logit (>0 means correct)."""
    t = logits[np.arange(len(y)), y]
    o = logits.copy()
    o[np.arange(len(y)), y] = -np.inf
    return t - o.max(1)


def spearman_rows(a, b):
    ra = rankdata(a, axis=1)
    rb = rankdata(b, axis=1)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    ok = den > 0
    return float(np.mean((ra * rb).sum(1)[ok] / den[ok]))


# ----------------------------------------------------------------------------- analyses
def robustness(model, mean, std, P, variants, y, logits0):
    pred0 = logits0.argmax(1)
    out = {}
    for v, Xv in zip(variants, P):
        lg = parts(model, (Xv - mean) / std)[0]
        p = lg.argmax(1)
        out[str(v)] = {'acc': float(np.mean(p == y)), 'prediction_change': float(np.mean(p != pred0)),
                       'correct_to_wrong': float(np.mean((pred0 == y) & (p != y))),
                       'wrong_to_correct': float(np.mean((pred0 != y) & (p == y)))}
    loc = [k for k in out if k.startswith('local')]
    out['local_mean'] = {m: float(np.mean([out[k][m] for k in loc])) for m in out[loc[0]]}
    return out


def path_swap(model, mean, std, P, variants, y):
    """2x2: (clean | perturbed representation z) x (clean | perturbed weights alpha).
    For clips that the full perturbation turns from correct to wrong, ask whether swapping in
    only the perturbed z, or only the perturbed alpha, is already enough to make it wrong."""
    _, zc, ac, _ = parts(model, (P[0] - mean) / std)
    res, agg = {}, {'n_flip': 0, 'z_only': 0, 'alpha_only': 0, 'both_alone': 0, 'neither_alone': 0,
                    'dmargin_z': [], 'dmargin_a': [], 'dmargin_full': []}
    with torch.no_grad():
        base = model.pool_and_classify(zc, ac).numpy()
        for v, Xv in zip(variants[1:], P[1:]):
            _, zp, ap, _ = parts(model, (Xv - mean) / std)
            full = model.pool_and_classify(zp, ap).numpy()
            zo = model.pool_and_classify(zp, ac).numpy()
            ao = model.pool_and_classify(zc, ap).numpy()
            flip = (base.argmax(1) == y) & (full.argmax(1) != y)
            zw, aw = zo.argmax(1) != y, ao.argmax(1) != y
            m0 = margin(base, y)
            r = {'n_flip': int(flip.sum()),
                 'z_only_sufficient': float(np.mean(zw[flip])) if flip.any() else None,
                 'alpha_only_sufficient': float(np.mean(aw[flip])) if flip.any() else None,
                 'mean_dmargin_z_only': float(np.mean(margin(zo, y) - m0)),
                 'mean_dmargin_alpha_only': float(np.mean(margin(ao, y) - m0)),
                 'mean_dmargin_full': float(np.mean(margin(full, y) - m0)),
                 'mean_alpha_tv': float(0.5 * np.abs(ap - ac).sum(1).mean().item())}
            if str(v).startswith('local'):
                j = int(str(v).split('_s')[1])
                r['mean_alpha_change_on_perturbed_segment'] = float((ap[:, j] - ac[:, j]).mean().item())
                agg['n_flip'] += int(flip.sum())
                agg['z_only'] += int((zw & ~aw)[flip].sum())
                agg['alpha_only'] += int((aw & ~zw)[flip].sum())
                agg['both_alone'] += int((aw & zw)[flip].sum())
                agg['neither_alone'] += int((~aw & ~zw)[flip].sum())
                agg['dmargin_z'].append(r['mean_dmargin_z_only'])
                agg['dmargin_a'].append(r['mean_dmargin_alpha_only'])
                agg['dmargin_full'].append(r['mean_dmargin_full'])
            res[str(v)] = r
    for k in ('dmargin_z', 'dmargin_a', 'dmargin_full'):
        agg[k] = float(np.mean(agg[k]))
    res['local_aggregate'] = agg
    return res


def attention_diagnostics(model, mean, std, Xraw, y):
    """Does attention go to the loudest segment, or to the segment that is useful on its own?"""
    Xs = (Xraw - mean) / std
    logits, z, a, contrib = parts(model, Xs)
    a = a.numpy()
    with torch.no_grad():   # each segment classified alone through the shared head
        alone = model.head(z).numpy().argmax(2)                       # (N, K)
    alone_ok = alone == y[:, None]
    loud = Xraw[:, :, LOGRMS]
    rows = np.arange(len(y))
    top_a, top_loud = a.argmax(1), loud.argmax(1)
    # causal check: drop one segment (renormalise the remaining weights) and re-classify
    def drop(sel):
        aa = a.copy()
        aa[rows, sel] = 0
        aa /= aa.sum(1, keepdims=True)
        with torch.no_grad():
            lg = model.pool_and_classify(z, torch.as_tensor(aa)).numpy()
        return float(np.mean(lg.argmax(1) == y))
    rng = np.random.default_rng(0)
    return {
        'entropy_mean': float(-(a * np.log(a + 1e-12)).sum(1).mean()), 'entropy_uniform': float(np.log(a.shape[1])),
        'max_weight_mean': float(a.max(1).mean()),
        'mean_weight_by_position': a.mean(0).tolist(),
        'spearman_alpha_vs_logrms': spearman_rows(a, loud),
        'spearman_alpha_vs_voiced_frac': spearman_rows(a, Xraw[:, :, VOICED]),
        'spearman_alpha_vs_logf0': spearman_rows(a, Xraw[:, :, LOGF0]),
        'top_attended_is_loudest': float(np.mean(top_a == top_loud)),
        'segment_alone_correct_top_attended': float(np.mean(alone_ok[rows, top_a])),
        'segment_alone_correct_loudest': float(np.mean(alone_ok[rows, top_loud])),
        'segment_alone_correct_any': float(np.mean(alone_ok)),
        'acc_full': float(np.mean(logits.argmax(1) == y)),
        'acc_drop_top_attended': drop(top_a),
        'acc_drop_loudest': drop(top_loud),
        'acc_drop_random': float(np.mean([drop(rng.integers(0, a.shape[1], len(y))) for _ in range(10)])),
    }


def taxonomy(pred, y, votes):
    """Failure taxonomy against audio-only listener votes (plurality; ties kept as sets)."""
    n = votes.sum(1)
    plural = votes == votes.max(1, keepdims=True)
    rows = np.arange(len(y))
    lis_actor = plural[rows, y]                       # listeners' plurality includes the actor label
    lis_pred = plural[rows, pred]
    support = votes[rows, y] / n
    cat = np.where(pred == y, 'correct',
          np.where(lis_actor & ~lis_pred, 'clear_miss',
          np.where(lis_pred, 'matches_listeners', 'neither')))
    out = {'listener_plurality_includes_actor': float(np.mean(lis_actor)),
           'model_acc_vs_actor': float(np.mean(pred == y)),
           'model_in_listener_plurality': float(np.mean(lis_pred)),
           'categories': {c: float(np.mean(cat == c)) for c in ('correct', 'clear_miss', 'matches_listeners', 'neither')}}
    errs = cat != 'correct'
    out['error_categories'] = {c: float(np.mean(cat[errs] == c)) for c in ('clear_miss', 'matches_listeners', 'neither')}
    bins = [0, .2, .4, .6, .8, 1.01]
    out['acc_by_listener_support'] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (support >= lo) & (support < hi)
        out['acc_by_listener_support'].append({'bin': [lo, min(hi, 1.0)], 'n': int(m.sum()),
                                               'model_acc': float(np.mean(pred[m] == y[m])) if m.any() else None})
    # per actor class: where do the errors go, and do listeners go there too?
    out['by_class'] = {}
    for c, name in enumerate(F.EMOTIONS):
        m = y == c
        e = m & (pred != y)
        out['by_class'][name] = {
            'n': int(m.sum()), 'model_acc': float(np.mean(pred[m] == c)),
            'listener_plurality_has_actor': float(np.mean(lis_actor[m])),
            'model_error_targets': {F.EMOTIONS[k]: int(np.sum(pred[e] == k)) for k in range(6) if k != c},
            'listener_votes_share': {F.EMOTIONS[k]: float(votes[m, k].sum() / n[m].sum()) for k in range(6)},
        }
    return out, cat, support


# ----------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, meta = load_data()
    groups = speaker_split(meta['speaker'])
    test = np.where(np.isin(meta['speaker'], groups['test']))[0]
    pz = np.load(ROOT / 'data' / 'hw3_perturbed.npz')
    P, variants = pz['P'], [str(v) for v in pz['variants']]
    assert np.array_equal(pz['test_idx'], test)
    yt, Xt = y[test], X[test]
    votes = audio_votes(meta['fname'][test])
    C, U = build_pairs(meta, test)
    pos = {g: i for i, g in enumerate(test)}
    Cl, Ul = np.vectorize(pos.get)(C), np.vectorize(pos.get)(U)

    res = {'variants': variants, 'n_test': int(len(test)), 'models': {}}
    preds = {}
    for pooling in POOLINGS:
        for seed in SEEDS:
            name = f'{pooling}_s{seed}'
            if not (RUNS / name / 'model.pt').exists():
                continue
            model, mean, std = load_model(pooling, seed)
            logits0 = parts(model, (Xt - mean) / std)[0]
            pred = logits0.argmax(1)
            preds[name] = pred
            m = json.loads((RUNS / name / 'metrics.json').read_text())
            r = {k: m[k] for k in ('best_epoch', 'n_params', 'train_seconds')}
            r.update({s: {k: m[s][k] for k in ('acc', 'macro_f1', 'loss')} for s in ('train', 'val', 'test')})
            r['paired'] = {'false_flip_U': float(np.mean(pred[Ul[:, 0]] != pred[Ul[:, 1]])),
                           'prediction_change_C': float(np.mean(pred[Cl[:, 0]] != pred[Cl[:, 1]])),
                           'strict_correct_flip_C': float(np.mean((pred[Cl[:, 0]] == yt[Cl[:, 0]])
                                                                  & (pred[Cl[:, 1]] == yt[Cl[:, 1]])))}
            r['robustness'] = robustness(model, mean, std, P, variants, yt, logits0)
            if model.pooling != 'concat':
                r['path_swap'] = path_swap(model, mean, std, P, variants, yt)
            else:   # where does the position-specific read-out put its weight?
                W = model.head.weight.detach().view(model.n_classes, model.k, -1)
                contrib = parts(model, (Xt - mean) / std)[3]
                r['readout_by_position'] = {
                    'weight_norm': W.norm(dim=(0, 2)).tolist(),
                    'mean_abs_contribution': np.abs(contrib).sum(2).mean(0).tolist(),
                    'mean_true_class_contribution': contrib[np.arange(len(yt)), :, yt].mean(0).tolist()}
            if model.pooling in ('attn', 'attn_rel'):
                r['attention'] = attention_diagnostics(model, mean, std, Xt, yt)
            r['taxonomy'] = taxonomy(pred, yt, votes)[0]
            res['models'][name] = r
            print(f"{name:12s} test {r['test']['acc']:.3f}  falseflipU {r['paired']['false_flip_U']:.3f}  "
                  f"global-6 change {r['robustness']['global_-6']['prediction_change']:.3f}  "
                  f"local change {r['robustness']['local_mean']['prediction_change']:.3f}", flush=True)

    # listener baseline: how often the audio-only listener plurality matches the actor label
    plural = votes == votes.max(1, keepdims=True)
    res['listeners'] = {'plurality_includes_actor': float(np.mean(plural[np.arange(len(yt)), yt])),
                        'strict_plurality_equals_actor': float(np.mean((votes.argmax(1) == yt)
                                                                       & (plural.sum(1) == 1))),
                        'mean_responses': float(votes.sum(1).mean())}
    np.savez_compressed(OUT / 'test_predictions.npz', test_idx=test, y=yt, votes=votes,
                        **{k: v for k, v in preds.items()})
    (OUT / 'analysis.json').write_text(json.dumps(res, indent=1))
    print('saved', OUT / 'analysis.json')


if __name__ == '__main__':
    main()
