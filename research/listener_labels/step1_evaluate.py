"""Step 1: evaluate the saved personal-reference models (A, B, D2, D3) against listener votes. No training.

    python research/listener_labels/step1_evaluate.py
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(HERE))
from listener_metrics import evaluate, actor_bootstrap  # noqa: E402

SEEDS = (7, 11, 23)
CONDITIONS = ('A', 'B', 'D2', 'D3')
KEYS = ('actor_correct', 'in_listener_plurality', 'ce_to_votes', 'brier_to_votes', 'jaccard', 'set_recall', 'set_size', 'max_prob')


def votes_for(fnames, path=ROOT / 'data/tabulatedVotes.csv'):
    rows = {r['fileName']: r for r in csv.DictReader(open(path)) if int(r['']) // 100000 == 1}
    v = np.array([[int(rows[Path(str(f)).stem][c]) for c in 'ADFHNS'] for f in fnames], float)
    return v / v.sum(1, keepdims=True)


def main():
    z = np.load(ROOT / 'hw3/personal_reference/results/test-predictions.npz', allow_pickle=True)
    y, spk = z['y'], z['speaker'].astype(str)
    q = votes_for(z['fname'])
    res = {}
    for c in CONDITIONS:
        runs = [evaluate(z[f'{c}_s{s}'], y, q, spk) for s in SEEDS]
        res[c] = {'mean': {k: float(np.mean([r[k] for r in runs])) for k in runs[0] if not isinstance(runs[0][k], dict)},
                  'groups': {g: {k: float(np.mean([r['groups'][g][k] for r in runs])) for k in runs[0]['groups'][g]}
                             for g in runs[0]['groups']},
                  'per_actor': {a: {k: float(np.mean([r['per_actor'][a][k] for r in runs])) for k in KEYS}
                                for a in runs[0]['per_actor']}}
    actors = sorted(res['A']['per_actor'])
    diffs = {}
    for a_, b_ in (('D2', 'A'), ('D3', 'A'), ('B', 'A')):
        diffs[f'{a_}_minus_{b_}'] = {k: actor_bootstrap([res[a_]['per_actor'][x][k] - res[b_]['per_actor'][x][k] for x in actors])
                                     for k in ('actor_correct', 'ce_to_votes', 'jaccard')}
    out = {'n_test': int(len(y)), 'conditions': {c: {'mean': v['mean'], 'groups': v['groups']} for c, v in res.items()},
           'paired_actor_differences': diffs}
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / 'step1.json').write_text(json.dumps(out, indent=1))
    show(out)


def show(out):
    cols = ('actor_correct', 'in_listener_plurality', 'ce_to_votes', 'brier_to_votes', 'jaccard', 'set_recall', 'set_size', 'ece_vs_actor',
            'awareness_rho_maxprob_vs_listener_share')
    print(f"{'':6s}" + ''.join(f'{c[:14]:>15s}' for c in cols))
    for c, v in out['conditions'].items():
        print(f'{c:6s}' + ''.join(f"{v['mean'][k]:15.3f}" for k in cols))
    print('\nmean max probability by listener consensus (low / mid / high):')
    for c, v in out['conditions'].items():
        print(f'{c:6s}', '  '.join(f"{g}: {v['groups'][g]['max_prob']:.3f} (acc {v['groups'][g]['actor_correct']:.3f}, CE {v['groups'][g]['ce_to_votes']:.3f})"
                                    for g in ('low', 'mid', 'high')))


if __name__ == '__main__':
    main()
