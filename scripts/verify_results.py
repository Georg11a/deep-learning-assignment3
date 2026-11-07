"""Read-only CPU verification of packaged checkpoints against stored predictions."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)
torch.serialization.add_safe_globals([np.ndarray, np.dtype, np._core.multiarray._reconstruct,
                                      np.dtypes.Float32DType, np.dtypes.Float64DType])


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def predict(model, x):
    model.eval()
    return torch.cat([model(b).softmax(1) for b in torch.tensor(x).split(512)]).numpy()


def compare(name, actual, saved):
    expected = saved if saved.ndim == 1 else saved.argmax(1)
    changed = int((actual.argmax(1) != expected).sum())
    assert changed == 0, f'{name}: {changed} changed predictions'
    if saved.ndim == 2:
        np.testing.assert_allclose(actual, saved, atol=2e-5, rtol=2e-5, err_msg=name)


def main():
    d = np.load(ROOT / 'data/features_K8.npz'); checked = 0
    pilot = module('pilot_models', ROOT / 'hw3/models.py')
    old = np.load(ROOT / 'hw3/results/test_predictions.npz')
    for kind in pilot.POOLINGS:
        for seed in (7, 11, 23):
            name = f'{kind}_s{seed}'
            ck = torch.load(ROOT / 'hw3/runs' / name / 'model.pt', map_location='cpu', weights_only=True)
            m = pilot.PooledSegmentModel(ck['d_in'], k=ck['k'], pooling=ck['pooling'], level_idx=ck['level_idx'])
            m.load_state_dict(ck['state_dict'])
            x = (d['X'][old['test_idx']] - ck['mean']) / ck['std']
            compare(name, predict(m, x), old[name]); checked += 1
    level = ROOT / 'hw3/level_tradeoff'
    sys.path.insert(0, str(level))
    exp = module('level_experiment', level / 'experiment.py')
    config = json.loads((level / 'results/config.json').read_text())
    assert sha(level / 'experiment.py') == config['code_sha256']
    assert sha(level / 'protocol-zh.txt') == config['protocol_sha256']
    assert sha(ROOT / 'data/features_K8.npz') == config['feature_sha256']
    old = np.load(level / 'results/test-predictions.npz'); norm = np.load(level / 'results/normalization.npz')
    x = (d['X'][old['idx']] - norm['mean']) / norm['std']
    for p in exp.PS:
        for seed in exp.SEEDS:
            name = f'C_p{p:g}_s{seed}'; m = exp.LevelModel(p)
            m.load_state_dict(torch.load(level / 'results' / f'{name}.pt', map_location='cpu', weights_only=True))
            compare(name, predict(m, x), old[name]); checked += 1
    personal = ROOT / 'hw3/personal_reference'
    exp = module('personal_experiment', personal / 'run.py')
    config = json.loads((personal / 'results/config.json').read_text())
    assert sha(personal / 'run.py') == config['code_sha256']
    assert sha(personal / 'protocol-zh.txt') == config['protocol_sha256']
    assert sha(ROOT / 'data/features_K8.npz') == config['features_sha256']
    assert sha(ROOT / 'data/tabulatedVotes.csv') == config['votes_sha256']
    groups, en, query, x, refs, mean, std = exp.prepare(d)
    exp.validate(d, groups, en, query, x, refs)
    old = np.load(personal / 'results/test-predictions.npz'); ii = query['test']
    np.testing.assert_array_equal(ii, old['idx'])
    frozen = json.loads((personal / 'results/frozen-before-test.json').read_text())
    assert sha(personal / 'results/val.json') == frozen['validation_sha256']
    complete = json.loads((personal / 'results/completed.json').read_text())
    assert sha(personal / 'results/test.json') == complete['test_sha256']
    for condition in exp.CONDITIONS:
        for seed in exp.SEEDS:
            name = f'{condition}_s{seed}'; checkpoint = personal / 'results' / f'{name}.pt'
            assert sha(checkpoint) == frozen['models'][name]
            m = exp.SegmentMLP(88, dropout=.5)
            m.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
            xp = exp.inputs(x[ii], d['speaker'][ii], refs, condition)
            compare(name, predict(m, xp), old[name]); checked += 1
            if condition in ('D2', 'D3'):
                for offset in range(1, 6):
                    xp = exp.inputs(x[ii], d['speaker'][ii], refs, condition, groups['test'], offset)
                    compare(f'{name}_wrong{offset}', predict(m, xp), old[f'{name}_wrong{offset}'])
    overlap = module('sentence_overlap', personal / 'sentence_overlap_check.py')
    recomputed, _ = overlap.compute(personal / 'results')
    saved_overlap = json.loads((personal / 'results/sentence-overlap-check.json').read_text())
    assert recomputed['manifest_sha256'] == saved_overlap['manifest_sha256']
    assert recomputed['prediction_sha256'] == saved_overlap['prediction_sha256']
    for subset, r in recomputed['subsets'].items():
        expected = saved_overlap['subsets'][subset]
        assert r['n_queries'] == expected['n_queries']
        for name, pair in r['paired'].items():
            np.testing.assert_allclose(pair['difference_pp'], expected['paired'][name]['difference_pp'], atol=1e-12)
            np.testing.assert_allclose(pair['actor_bootstrap_95_pp'], expected['paired'][name]['actor_bootstrap_95_pp'], atol=1e-12)
    print(json.dumps({'checkpoints_verified': checked, 'changed_top1_predictions': 0,
                      'all_30_personal_mismatch_evaluations_verified': True,
                      'personal_enrollment_checks_passed': True,
                      'posthoc_sentence_overlap_check_verified': True,
                      'frozen_source_data_and_model_hashes_passed': True}, indent=2))


if __name__ == '__main__':
    main()
