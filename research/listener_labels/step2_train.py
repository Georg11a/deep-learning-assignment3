"""Step 2: same D2 input as the personal-reference study, three training targets (see protocol-zh.txt).

  L-A  actor label (hard)
  L-B  listener plurality (ties split evenly; minority votes dropped)
  L-C  full listener vote distribution (soft)

Checkpoint = earliest epoch with the lowest validation cross-entropy to the condition's own target.
A replication run (L-A with the personal-reference accuracy rule) checks the pipeline against saved D2 models.

    python research/listener_labels/step2_train.py            # trains 9 models + 3 replication models, then scores
"""
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from training.model import SegmentMLP  # noqa: E402
from listener_metrics import evaluate, actor_bootstrap  # noqa: E402
from step1_evaluate import votes_for  # noqa: E402

spec = importlib.util.spec_from_file_location('pr', ROOT / 'hw3/personal_reference/run.py')
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

SEEDS = (7, 11, 23)
LABELS = ('L-A', 'L-B', 'L-C')
OUT = HERE / 'results'
KEYS = ('actor_correct', 'in_listener_plurality', 'ce_to_votes', 'brier_to_votes', 'jaccard', 'set_recall', 'set_size', 'max_prob')


def targets(label, y, q):
    if label == 'L-A':
        return np.eye(6)[y]
    if label == 'L-B':
        top = (q == q.max(1, keepdims=True)).astype(float)
        return top / top.sum(1, keepdims=True)
    return q


def fit(label, seed, x, t, query, rule='own_target_ce'):
    """Same order of RNG use as hw3/personal_reference/run.py: manual_seed, model init, per-epoch randperm."""
    torch.manual_seed(seed)
    m = SegmentMLP(88, dropout=.5)
    opt = torch.optim.SGD(m.parameters(), lr=.02, momentum=.9, weight_decay=.005)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 60)
    xt = torch.tensor(x); tt = torch.tensor(t, dtype=torch.float32)
    tr = torch.tensor(query['train']); va = torch.tensor(query['val'])
    yv = tt[va].argmax(1)
    hard = bool(((tt == 0) | (tt == 1)).all() and (tt.sum(1) == 1).all())
    best, history = None, []
    for epoch in range(1, 61):
        m.train()
        for b in torch.randperm(len(tr)).split(64):
            ii = tr[b]
            lg = m(xt[ii])
            if hard:   # identical call to the personal-reference code, so the replication is exact
                loss = torch.nn.functional.cross_entropy(lg, tt[ii].argmax(1))
            else:
                loss = -(tt[ii] * torch.log_softmax(lg, 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step(); m.eval()
        with torch.no_grad():
            lg = m(xt[va])
            ce = float(-(tt[va] * torch.log_softmax(lg, 1)).sum(1).mean())
            acc = float((lg.argmax(1) == yv).float().mean())
        history.append({'epoch': epoch, 'val_own_ce': ce, 'val_acc_vs_target_top': acc})
        score = ce if rule == 'own_target_ce' else -acc
        if best is None or score < best[0]:
            best = (score, epoch, {k: v.clone() for k, v in m.state_dict().items()})
    final = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best[2]); m.eval()
    return m, best[1], final, history


@torch.no_grad()
def probs(m, x):
    m.eval()
    return torch.softmax(m(torch.tensor(x)), 1).numpy()


def summarize_runs(runs):
    return {'mean': {k: float(np.mean([r[k] for r in runs])) for k in runs[0] if not isinstance(runs[0][k], dict)},
            'groups': {g: {k: float(np.mean([r['groups'][g][k] for r in runs])) for k in runs[0]['groups'][g]} for g in runs[0]['groups']},
            'per_actor': {a: {k: float(np.mean([r['per_actor'][a][k] for r in runs])) for k in KEYS} for a in runs[0]['per_actor']}}


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    d = np.load(ROOT / 'data/features_K8.npz')
    groups, en, query, x, refs, mean, std = pr.prepare(d)
    pr.validate(d, groups, en, query, x, refs)
    xin = pr.inputs(x, d['speaker'], refs, 'D2')
    y = np.array([pr.EMOTIONS.index(e) for e in d['emotion']])
    q = votes_for(d['fname'])
    prior = q[query['train']].mean(0)
    saved = np.load(ROOT / 'hw3/personal_reference/results/test-predictions.npz', allow_pickle=True)
    assert np.array_equal(saved['idx'], query['test'])

    models, info, finals = {}, {}, {}
    t0 = time.time()
    for label in LABELS:
        t = targets(label, y, q)
        for s in SEEDS:
            m, ep, final, hist = fit(label, s, xin, t, query)
            models[(label, s)] = m; finals[(label, s)] = final
            info[f'{label}_s{s}'] = {'best_epoch': ep, 'history': hist}
            print(f'{label} seed {s}: best epoch {ep}  ({time.time() - t0:.0f}s)', flush=True)
    # replication: L-A with the personal-reference rule must reproduce saved D2 predictions
    repl = {}
    yt = y[query['test']]
    for s in SEEDS:
        m, ep, _, _ = fit('L-A', s, xin, targets('L-A', y, q), query, rule='val_acc')
        p = probs(m, xin[query['test']])
        repl[s] = {'best_epoch': ep, 'max_abs_prob_diff_vs_saved_D2': float(np.abs(p - saved[f'D2_s{s}']).max()),
                   'top1_changed_vs_saved_D2': int((p.argmax(1) != saved[f'D2_s{s}'].argmax(1)).sum()),
                   'test_acc_rerun': float((p.argmax(1) == yt).mean()),
                   'test_acc_saved_D2': float((saved[f'D2_s{s}'].argmax(1) == yt).mean())}
    print('replication of saved D2:', repl, flush=True)

    def score(stage):
        ii = query[stage]; ys, qs, sp = y[ii], q[ii], d['speaker'][ii].astype(str)
        res = {lab: summarize_runs([evaluate(probs(models[(lab, s)], xin[ii]), ys, qs, sp) for s in SEEDS]) for lab in LABELS}
        for lab in LABELS:   # final-epoch sensitivity check
            runs = []
            for s in SEEDS:
                mm = SegmentMLP(88, dropout=.5); mm.load_state_dict(finals[(lab, s)])
                runs.append(evaluate(probs(mm, xin[ii]), ys, qs, sp))
            res[lab + '@ep60'] = summarize_runs(runs)
        res['P0 uniform'] = summarize_runs([evaluate(np.full((len(ii), 6), 1 / 6), ys, qs, sp)])
        res['P1 vote prior'] = summarize_runs([evaluate(np.tile(prior, (len(ii), 1)), ys, qs, sp)])
        return res

    val = score('val')
    (OUT / 'step2-val.json').write_text(json.dumps({k: {'mean': v['mean'], 'groups': v['groups']} for k, v in val.items()}, indent=1))
    (OUT / 'step2-frozen-before-test.json').write_text(json.dumps({
        'utc': datetime.now(timezone.utc).isoformat(), 'rule': 'All conditions reported; no test-based choice.',
        'best_epochs': {k: v['best_epoch'] for k, v in info.items()}, 'replication': repl}, indent=1))
    test = score('test')
    # collapse check and paired actor differences on test
    ii = query['test']
    kl_to_prior = {}
    for lab in LABELS:
        ps = np.mean([probs(models[(lab, s)], xin[ii]) for s in SEEDS], 0)
        kl_to_prior[lab] = float(np.mean((ps * (np.log(ps + 1e-9) - np.log(prior))).sum(1)))
    actors = sorted(test['L-A']['per_actor'])
    diffs = {}
    for a_, b_ in (('L-C', 'L-A'), ('L-C', 'L-B'), ('L-B', 'L-A')):
        diffs[f'{a_} minus {b_}'] = {k: actor_bootstrap([test[a_]['per_actor'][x][k] - test[b_]['per_actor'][x][k] for x in actors])
                                     for k in ('actor_correct', 'in_listener_plurality', 'ce_to_votes', 'jaccard', 'max_prob')}
    diffs['L-C minus P1 vote prior'] = {'ce_to_votes': actor_bootstrap(
        [test['L-C']['per_actor'][x]['ce_to_votes'] - test['P1 vote prior']['per_actor'][x]['ce_to_votes'] for x in actors])}
    out = {'test': {k: {'mean': v['mean'], 'groups': v['groups']} for k, v in test.items()},
           'mean_kl_prediction_to_vote_prior': kl_to_prior, 'paired_actor_differences': diffs,
           'replication_of_saved_D2': repl, 'best_epochs': {k: v['best_epoch'] for k, v in info.items()}}
    # save every test probability, plus the secondary diagnostics named in the protocol
    preds = {'idx': ii, 'y': y[ii], 'speaker': d['speaker'][ii], 'fname': d['fname'][ii], 'votes': q[ii]}
    E = pr.EMOTIONS
    sec = {}
    loud = {s_: float(refs[s_][84] * std[84] + mean[84]) for s_ in refs}
    from scipy.stats import spearmanr
    for lab in LABELS:
        rates, shares, recall, rhos = [], [], [], []
        for s in SEEDS:
            p = probs(models[(lab, s)], xin[ii]); preds[f'{lab}_s{s}'] = p
            pred = p.argmax(1); yt = y[ii]; sp = d['speaker'][ii].astype(str)
            neu = yt == 4
            rates.append(float((pred[neu] == 5).mean()))
            shares.append(np.bincount(pred, minlength=6) / len(pred))
            recall.append([float((pred[yt == c] == c).mean()) for c in range(6)])
            per = [(loud[a], float((pred[neu & (sp == a)] == 5).mean())) for a in sorted(set(sp))]
            rhos.append(float(spearmanr(*zip(*per)).statistic))
        sec[lab] = {'neutral_to_sad': float(np.mean(rates)), 'rho_enroll_loudness_vs_neutral_to_sad': float(np.mean(rhos)),
                    'pred_share': dict(zip(E, np.mean(shares, 0).round(4).tolist())),
                    'actor_recall': dict(zip(E, np.mean(recall, 0).round(4).tolist()))}
    out['secondary'] = sec
    np.savez_compressed(OUT / 'step2-test-predictions.npz', **preds)
    (OUT / 'step2-test.json').write_text(json.dumps(out, indent=1))
    show(out)


def show(out):
    cols = ('actor_correct', 'in_listener_plurality', 'ce_to_votes', 'brier_to_votes', 'jaccard', 'set_size', 'ece_vs_actor',
            'awareness_rho_maxprob_vs_listener_share')
    print(f"\n{'':16s}" + ''.join(f'{c[:13]:>14s}' for c in cols))
    for k, v in out['test'].items():
        print(f'{k:16s}' + ''.join(f"{v['mean'][c]:14.3f}" for c in cols))
    print('\nmax prob by consensus (low/mid/high):')
    for k in ('L-A', 'L-B', 'L-C'):
        g = out['test'][k]['groups']
        print(f'{k:6s}', '  '.join(f"{n}: maxp {g[n]['max_prob']:.3f} acc {g[n]['actor_correct']:.3f} CE {g[n]['ce_to_votes']:.3f}" for n in ('low', 'mid', 'high')))
    print('\nKL(prediction || vote prior):', {k: round(v, 3) for k, v in out['mean_kl_prediction_to_vote_prior'].items()})
    for k, v in out['paired_actor_differences'].items():
        print(k, {m: (round(x['mean'], 3), [round(c, 3) for c in x['ci95']]) for m, x in v.items()})


if __name__ == '__main__':
    main()
