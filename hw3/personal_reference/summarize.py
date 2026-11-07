"""Summarize the frozen personal-reference study; no fitting or model selection."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

SEEDS = (7, 11, 23)
CONDITIONS = ('A', 'B', 'D2', 'D3')


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))


def interval(values):
    values = np.asarray(values)
    draws = np.random.default_rng(20261108).integers(0, len(values), (5000, len(values)))
    return {'mean_pp': float(values.mean() * 100),
            'actor_bootstrap_95_pp': (np.quantile(values[draws].mean(1), [.025, .975]) * 100).tolist(),
            'n_actors': len(values), 'bootstrap_repetitions': 5000}


def correlation(x, y):
    def rho(a, b):
        aa = rankdata(a, axis=-1); bb = rankdata(b, axis=-1)
        aa -= aa.mean(axis=-1, keepdims=True); bb -= bb.mean(axis=-1, keepdims=True)
        den = np.sqrt((aa**2).sum(-1) * (bb**2).sum(-1))
        return np.divide((aa*bb).sum(-1), den, out=np.full(den.shape, np.nan), where=den > 0)
    draws = np.random.default_rng(20261109).integers(0, len(x), (5000, len(x)))
    bootstrap = rho(x[draws], y[draws]); valid = bootstrap[np.isfinite(bootstrap)]
    point = float(rho(x, y))
    return {'rho': point if np.isfinite(point) else None,
            'actor_bootstrap_95': np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
            'valid_bootstrap_draws': len(valid), 'n_actors': len(x)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, default=Path(__file__).parent / 'results')
    args = ap.parse_args(); out = args.out
    assert (out / 'completed.json').exists(), 'Wait for the frozen experiment to finish.'
    tables = {}; all_intervals = {}; neutral_rows = []
    norm = np.load(out / 'normalization.npz')
    raw_ref = norm['references'] * norm['std'] + norm['mean']
    loudness = dict(zip(norm['speakers'], raw_ref[:, 84]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for stage, ax in zip(('val', 'test'), axes):
        data = json.loads((out / f'{stage}.json').read_text())
        actors = sorted(data['A_s7']['matched']['per_actor'])
        rows = []; per = {}
        for condition in CONDITIONS:
            rr = [data[f'{condition}_s{s}']['matched'] for s in SEEDS]
            row = {'condition': condition}
            for metric in ('accuracy', 'actor_macro_accuracy', 'macro_f1', 'neutral_to_sad', 'listener_clear_accuracy'):
                scale = 1 if metric == 'macro_f1' else 100
                row[metric] = float(np.mean([r[metric] for r in rr]) * scale)
                row[metric + '_seed_sd'] = float(np.std([r[metric] for r in rr]) * scale)
            row.update({f'recall_{c}': float(np.mean([r['recall'][c] for r in rr]) * 100) for c in ('ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD')})
            rows.append(row)
            per[condition] = np.array([np.mean([r['per_actor'][a]['accuracy'] for r in rr]) for a in actors])
        estimates = {'D3_minus_B': interval(per['D3'] - per['B']),
                     'D3_minus_A': interval(per['D3'] - per['A']),
                     'D3_minus_D2': interval(per['D3'] - per['D2'])}
        for condition in ('D2', 'D3'):
            wrong = np.array([np.mean([data[f'{condition}_s{s}']['mismatched'][str(o)]['per_actor'][a]['accuracy']
                                      for s in SEEDS for o in range(1, 6)]) for a in actors])
            estimates[f'{condition}_matched_minus_mismatched'] = interval(per[condition] - wrong)
            rows[CONDITIONS.index(condition)]['wrong_reference_actor_accuracy'] = float(wrong.mean() * 100)
        x = np.array([loudness[a] for a in actors]); correlations = {}
        for condition in CONDITIONS:
            rates = np.array([np.mean([data[f'{condition}_s{s}']['matched']['per_actor'][a]['neutral_to_sad'] for s in SEEDS]) for a in actors])
            correlations[condition] = correlation(x, rates)
            for a, xx, yy in zip(actors, x, rates):
                neutral_rows.append({'split': stage, 'condition': condition, 'actor': a,
                                     'registration_logrms': float(xx), 'neutral_to_sad': float(yy)})
        listen = np.array([data['A_s7']['matched']['per_actor'][a]['neutral_listener_sad'] for a in actors])
        correlations['listener_sad_vote_fraction'] = correlation(x, listen)
        all_intervals[stage] = {'paired_actor_differences': estimates, 'loudness_neutral_sad_correlations': correlations}
        tables[stage] = rows
        fieldnames = sorted(set(k for row in rows for k in row))
        with open(out / f'{stage}-summary.csv', 'w') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
        xx = np.arange(4)
        ax.bar(xx, [r['actor_macro_accuracy'] for r in rows], color=['#7b8793', '#2b8c89', '#8d7db2', '#e29839'], width=.64)
        ax.errorbar(xx, [r['actor_macro_accuracy'] for r in rows], yerr=[r['actor_macro_accuracy_seed_sd'] for r in rows], fmt='none', color='black', capsize=4)
        for j, condition in enumerate(CONDITIONS):
            ax.scatter(j + np.linspace(-.12, .12, 3), [data[f'{condition}_s{s}']['matched']['actor_macro_accuracy'] * 100 for s in SEEDS], c='black', s=14, zorder=3)
            ax.text(j, rows[j]['actor_macro_accuracy'] + 2, f"{rows[j]['actor_macro_accuracy']:.1f}", ha='center', fontsize=9)
        for j in (2, 3):
            ax.scatter(j, rows[j]['wrong_reference_actor_accuracy'], marker='x', color='#9f2020', s=65, zorder=4, label='Wrong reference (5 mappings)' if j == 2 else None)
        ax.set(xticks=xx, xticklabels=['A\nOriginal', 'B\nUtterance', 'D2\nOwn level', 'D3\n+ Own pitch'],
               ylabel='Equal-weight actor accuracy (%)', title=f'{stage.title()}: enrollment clips excluded', ylim=(0, 75))
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True); ax.legend(loc='lower left', fontsize=8)
    fig.suptitle('Does a personal reference improve recognition?', fontsize=14)
    fig.text(.5, .012, 'Bars: 3-seed mean. Error bars: seed SD (not a confidence interval). Same query set for all conditions.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .94)); fig.savefig(out / 'personal-reference.png', dpi=180); plt.close(fig)
    write_json(out / 'summary.json', tables); write_json(out / 'intervals.json', all_intervals)
    with open(out / 'neutral-diagnostic.csv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(neutral_rows[0])); w.writeheader(); w.writerows(neutral_rows)
    # All query predictions, rather than a hand-picked success gallery.
    pred = np.load(out / 'test-predictions.npz')
    with open(out / 'test-cases.csv', 'w') as f:
        fields = ['filename', 'actor', 'actor_label'] + [f'{c}_s{s}' for c in CONDITIONS for s in SEEDS]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        labels = ('ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD')
        for i, name in enumerate(pred['fname']):
            row = {'filename': str(name), 'actor': str(pred['speaker'][i]), 'actor_label': labels[pred['y'][i]]}
            row.update({f'{c}_s{s}': labels[pred[f'{c}_s{s}'][i].argmax()] for c in CONDITIONS for s in SEEDS})
            w.writerow(row)
    print(json.dumps({'tables': tables, 'intervals': all_intervals}, indent=2))


if __name__ == '__main__':
    main()
