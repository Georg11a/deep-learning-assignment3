"""Post-hoc sentence-disjoint evaluation of saved predictions; no fitting.

This independently implements the review suggestion. The externally mentioned
script was not available in the local workspace. Enrollment wording is defined
per actor from the frozen manifest, not inferred from model results.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

SEEDS = (7, 11, 23)
CONDITIONS = ('A', 'B', 'D2', 'D3')


def paired_interval(values):
    values = np.asarray(values, dtype=float)
    draws = np.random.default_rng(20261108).integers(0, len(values), (5000, len(values)))
    return {'difference_pp': float(values.mean() * 100),
            'actor_bootstrap_95_pp': (np.quantile(values[draws].mean(1), [.025, .975]) * 100).tolist(),
            'n_actors': len(values), 'bootstrap_repetitions': 5000,
            'bootstrap_seed': 20261108}


def compute(out):
    manifest_path = out / 'manifest.json'; prediction_path = out / 'test-predictions.npz'
    manifest = json.loads(manifest_path.read_text())
    saved = np.load(prediction_path)
    names = saved['fname']; actors = saved['speaker']; y = saved['y']
    assert len(set(names)) == len(names)
    np.testing.assert_array_equal(names, np.array(manifest['queries']['test']))
    assert set(names).isdisjoint(manifest['enrollment'])
    enrollment_sentences = {}
    for name in manifest['enrollment']:
        actor, sentence, emotion, _ = Path(name).stem.split('_')
        assert emotion == 'NEU'
        enrollment_sentences.setdefault(actor, set()).add(sentence)
    shared = np.array([Path(str(n)).stem.split('_')[1] in enrollment_sentences[str(a)]
                       for n, a in zip(names, actors)])
    masks = {'all_queries': np.ones(len(y), dtype=bool),
             'shared_sentence': shared, 'nonshared_sentence': ~shared}
    assert not np.any(masks['shared_sentence'] & masks['nonshared_sentence'])
    assert np.all(masks['shared_sentence'] | masks['nonshared_sentence'])
    correctness = {c: np.stack([saved[f'{c}_s{s}'].argmax(1) == y for s in SEEDS]) for c in CONDITIONS}
    result = {'scope': 'post-hoc saved-prediction sensitivity check; no retraining or new independent test',
              'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'prediction_sha256': hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
              'estimand': 'Equal-weight actor accuracy; average seeds within actor, then paired actor bootstrap.',
              'subsets': {}}
    rows = []
    for label, mask in masks.items():
        unique = np.unique(actors[mask]); per = {}; metrics = {}
        for c in CONDITIONS:
            per[c] = np.array([correctness[c][:, mask & (actors == a)].mean() for a in unique])
            metrics[c] = {'actor_accuracy_percent': float(per[c].mean() * 100),
                          'sample_accuracy_percent': float(correctness[c][:, mask].mean() * 100)}
            for a, acc in zip(unique, per[c]):
                rows.append({'subset': label, 'actor': str(a), 'condition': c,
                             'n_queries': int((mask & (actors == a)).sum()), 'accuracy_percent': float(acc * 100)})
        result['subsets'][label] = {'n_queries': int(mask.sum()), 'n_actors': len(unique),
                                   'conditions': metrics,
                                   'paired': {f'{c}_minus_B': paired_interval(per[c] - per['B']) for c in ('D2', 'D3')}}
    # The partition must recover each seed's total number of correct answers.
    for c, values in correctness.items():
        np.testing.assert_array_equal(values[:, shared].sum(1) + values[:, ~shared].sum(1), values.sum(1))
    # Agreement with the original all-query primary estimator is a useful audit.
    previous = json.loads((out / 'intervals.json').read_text())['test']['paired_actor_differences']['D3_minus_B']
    np.testing.assert_allclose(result['subsets']['all_queries']['paired']['D3_minus_B']['difference_pp'], previous['mean_pp'], atol=1e-12)
    result['checks_passed'] = ['manifest_prediction_alignment', 'unique_queries', 'no_enrollment_query_overlap',
                               'complete_disjoint_sentence_partition', 'per_seed_correct_count_reconstruction',
                               'original_actor_average_estimator_reproduced']
    return result, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path(__file__).parent / 'results')
    args = ap.parse_args(); result, rows = compute(args.out)
    (args.out / 'sentence-overlap-check.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    with open(args.out / 'sentence-overlap-per-actor.csv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps({k: {'n': v['n_queries'], 'paired': v['paired']} for k, v in result['subsets'].items()}, indent=2))


if __name__ == '__main__':
    main()
