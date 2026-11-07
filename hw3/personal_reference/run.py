"""Fixed-budget enrollment study; protocol-zh.txt is frozen before fitting."""
import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.model import SegmentMLP
from training.train import speaker_split

HERE = Path(__file__).resolve().parent
SEEDS = (7, 11, 23)
CONDITIONS = ('A', 'B', 'D2', 'D3')
EMOTIONS = ('ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD')
COLS = {'A': [], 'B': [0, 84], 'D2': [0, 84], 'D3': [0, 84, 80]}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def prepare(data):
    """Metadata-only enrollment selection; query features cannot change enrollment IDs."""
    speakers = data['speaker']
    unique = np.unique(speakers)
    groups = speaker_split(speakers)
    common = sorted(set.intersection(*[set(data['sentence'][(speakers == s) & (data['emotion'] == 'NEU')]) for s in unique]))
    np.random.default_rng(20261107).shuffle(common)
    sentences = common[:3]
    assert sentences == ['DFA', 'TAI', 'IOM'], 'Enrollment metadata changed; review protocol before running.'
    en = np.flatnonzero(np.isin(data['sentence'], sentences) & (data['emotion'] == 'NEU'))
    assert all(np.sum(speakers[en] == s) == 3 for s in unique)
    query = {g: np.flatnonzero(np.isin(speakers, ss) & ~np.isin(np.arange(len(speakers)), en)) for g, ss in groups.items()}
    assert set(groups['train']).isdisjoint(groups['val'])
    assert set(groups['train']).isdisjoint(groups['test'])
    assert set(groups['val']).isdisjoint(groups['test'])
    assert all(not np.intersect1d(en, q).size for q in query.values())
    raw = data['X'].astype(np.float32)
    flat = raw[query['train']].reshape(-1, 88)
    mean, std = flat.mean(0), flat.std(0) + 1e-6
    x = (raw - mean) / std
    refs = {str(s): x[en[speakers[en] == s]].mean(axis=(0, 1)) for s in unique}
    return groups, en, query, x, refs, mean, std


def inputs(x, speakers, refs, condition, group=None, offset=0):
    out = x.copy()
    cols = COLS[condition]
    if condition == 'B':
        out[:, :, cols] -= out[:, :, cols].mean(1, keepdims=True)
    elif condition in ('D2', 'D3'):
        mapping = {s: s for s in refs}
        if offset:
            order = sorted(group)
            assert 0 < offset < len(order)
            mapping.update({s: order[(i + offset) % len(order)] for i, s in enumerate(order)})
        r = np.stack([refs[mapping[str(s)]][cols] for s in speakers])
        out[:, :, cols] -= r[:, None, :]
    return out


def validate(data, groups, en, query, x, refs):
    speakers = data['speaker']
    assert np.isfinite(x).all()
    assert len(en) == 273 and sum(map(len, query.values())) + len(en) == len(x)
    for condition in CONDITIONS:
        result = inputs(x, speakers, refs, condition)
        unchanged = sorted(set(range(88)) - set(COLS[condition]))
        np.testing.assert_array_equal(result[:, :, unchanged], x[:, :, unchanged])
        if condition == 'B':
            np.testing.assert_allclose(result[:, :, COLS['B']].mean(1), 0, atol=2e-5)
    # Mutating an evaluation query must not modify registration statistics.
    ii = query['test'][0]
    altered = x.copy(); altered[ii] += 100
    s = str(speakers[ii]); original = x[en[speakers[en] == s]].mean(axis=(0, 1))
    np.testing.assert_array_equal(altered[en[speakers[en] == s]].mean(axis=(0, 1)), original)
    # A change to one person's reference has no effect on another person's inputs.
    modified = {k: v.copy() for k, v in refs.items()}; modified[s][0] += 10
    orig = inputs(x, speakers, refs, 'D3'); changed = inputs(x, speakers, modified, 'D3')
    np.testing.assert_array_equal(orig[speakers != s], changed[speakers != s])
    for group in groups.values():
        order = sorted(group)
        for offset in range(1, 6):
            assert all(a != order[(i + offset) % len(order)] for i, a in enumerate(order))
    return {'passed': True, 'checks': ['speaker_disjoint', 'enrollment_query_disjoint', 'three_enrollment_each',
            'unchanged_other_columns', 'B_zero_segment_mean', 'query_cannot_change_reference',
            'reference_is_person_specific', 'wrong_reference_no_self_match']}


@torch.no_grad()
def predict(model, x):
    model.eval()
    return torch.cat([model(b).softmax(1) for b in torch.tensor(x).split(512)]).numpy()


def metrics(prob, y, speakers, votes):
    pred = prob.argmax(1)
    cm = confusion_matrix(y, pred, labels=range(6))
    support = votes[np.arange(len(y)), y] / votes.sum(1)
    clear = support >= .6
    def optional_mean(a):
        return float(a.mean()) if len(a) else None
    per = {}
    for s in np.unique(speakers):
        m = speakers == s; neu = m & (y == 4)
        per[str(s)] = {'n': int(m.sum()), 'accuracy': float((pred[m] == y[m]).mean()),
                       'n_neutral': int(neu.sum()), 'neutral_to_sad': optional_mean(pred[neu] == 5),
                       'neutral_listener_sad': optional_mean(votes[neu, 5] / votes[neu].sum(1))}
    return {'accuracy': float((pred == y).mean()),
            'actor_macro_accuracy': float(np.mean([r['accuracy'] for r in per.values()])),
            'macro_f1': float(f1_score(y, pred, average='macro', labels=range(6), zero_division=0)),
            'recall': dict(zip(EMOTIONS, (cm.diagonal() / cm.sum(1)).tolist())),
            'confusion': cm.tolist(), 'neutral_to_sad': optional_mean(pred[y == 4] == 5),
            'listener_clear_n': int(clear.sum()), 'listener_clear_accuracy': optional_mean(pred[clear] == y[clear]),
            'per_actor': per}


def fit(condition, seed, x, y, query, out):
    torch.manual_seed(seed)
    m = SegmentMLP(88, dropout=.5)
    opt = torch.optim.SGD(m.parameters(), lr=.02, momentum=.9, weight_decay=.005)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 60)
    xt = torch.tensor(x); yt = torch.tensor(y)
    tr = torch.tensor(query['train']); va = torch.tensor(query['val'])
    best = -1; history = []; started = time.monotonic()
    for epoch in range(1, 61):
        m.train(); total = 0; correct = 0
        for b in torch.randperm(len(tr)).split(64):
            ii = tr[b]; lg = m(xt[ii]); loss = F.cross_entropy(lg, yt[ii])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b); correct += int((lg.argmax(1) == yt[ii]).sum())
        sched.step(); m.eval()
        with torch.no_grad():
            lg = m(xt[va]); acc = float((lg.argmax(1) == yt[va]).float().mean())
            vl = float(F.cross_entropy(lg, yt[va]))
        history.append({'epoch': epoch, 'train_loss': total / len(tr), 'train_accuracy': correct / len(tr),
                        'val_loss': vl, 'val_accuracy': acc})
        if acc > best:
            best = acc; best_epoch = epoch; state = {k: v.clone() for k, v in m.state_dict().items()}
    name = f'{condition}_s{seed}'; m.load_state_dict(state); m.eval()
    torch.save(state, out / f'{name}.pt')
    save(out / f'{name}.json', {'condition': condition, 'seed': seed, 'history': history,
         'best_epoch': best_epoch, 'parameters': sum(p.numel() for p in m.parameters()),
         'seconds': time.monotonic() - started})
    print(name, 'best validation', round(best, 4), 'epoch', best_epoch, flush=True)
    return m


def vote_table(path, names):
    with open(path) as f:
        rows = {r['fileName']: r for r in csv.DictReader(f) if int(r['']) // 100000 == 1}
    v = np.array([[int(rows[Path(str(n)).stem][c]) for c in 'ADFHNS'] for n in names])
    assert np.all(v.sum(1) > 0)
    return v


def score(stage, models, d, x, refs, query, groups, y, votes, out):
    ii = query[stage]; speakers = d['speaker'][ii]; ys = y[ii]; vs = votes[ii]
    result = {}; probs = {'idx': ii, 'y': ys, 'speaker': speakers, 'fname': d['fname'][ii]}
    for name, model in models.items():
        condition = name.split('_')[0]
        xp = inputs(x[ii], speakers, refs, condition)
        prob = predict(model, xp); probs[name] = prob
        r = {'matched': metrics(prob, ys, speakers, vs)}
        if condition in ('D2', 'D3'):
            r['mismatched'] = {}
            for offset in range(1, 6):
                xp = inputs(x[ii], speakers, refs, condition, groups[stage], offset)
                p = predict(model, xp); probs[f'{name}_wrong{offset}'] = p
                r['mismatched'][str(offset)] = metrics(p, ys, speakers, vs)
        result[name] = r
    save(out / f'{stage}.json', result)
    np.savez_compressed(out / f'{stage}-predictions.npz', **probs)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', type=Path, default=ROOT / 'data/features_K8.npz')
    ap.add_argument('--votes', type=Path, default=ROOT / 'data/tabulatedVotes.csv')
    ap.add_argument('--out', type=Path, default=HERE / 'results')
    ap.add_argument('--check-only', action='store_true')
    a = ap.parse_args(); torch.set_num_threads(1)
    d = np.load(a.features); groups, en, query, x, refs, mean, std = prepare(d)
    checks = validate(d, groups, en, query, x, refs)
    print('Checks:', checks, 'query counts:', {k: len(v) for k, v in query.items()}, flush=True)
    if a.check_only:
        return
    if a.out.exists():
        raise FileExistsError('Refusing to overwrite results; choose a new --out.')
    a.out.mkdir(parents=True)
    config = {'started_utc': now(), 'protocol_sha256': digest(HERE / 'protocol-zh.txt'),
              'code_sha256': digest(__file__), 'features_sha256': digest(a.features),
              'votes_sha256': digest(a.votes), 'features': str(a.features.resolve()),
              'conditions': CONDITIONS, 'seeds': SEEDS, 'groups': groups,
              'query_counts': {k: len(v) for k, v in query.items()},
              'torch': torch.__version__, 'numpy': np.__version__, 'primary_condition': 'D3',
              'recipe': {'epochs': 60, 'batch': 64, 'lr': .02, 'momentum': .9,
                         'weight_decay': .005, 'dropout': .5, 'scheduler': 'cosine',
                         'checkpoint': 'earliest maximal validation accuracy'}}
    save(a.out / 'config.json', config); save(a.out / 'checks.json', checks)
    save(a.out / 'manifest.json', {'enrollment': d['fname'][en].tolist(),
         'queries': {k: d['fname'][v].tolist() for k, v in query.items()},
         'enrollment_sentences': ['DFA', 'TAI', 'IOM']})
    np.savez_compressed(a.out / 'normalization.npz', mean=mean, std=std,
                        speakers=np.array(sorted(refs)), references=np.stack([refs[s] for s in sorted(refs)]))
    y = np.array([EMOTIONS.index(e) for e in d['emotion']])
    votes = vote_table(a.votes, d['fname'])
    models = {}
    for condition in CONDITIONS:
        xp = inputs(x, d['speaker'], refs, condition)
        for seed in SEEDS:
            models[f'{condition}_s{seed}'] = fit(condition, seed, xp, y, query, a.out)
    score('val', models, d, x, refs, query, groups, y, votes, a.out)
    save(a.out / 'frozen-before-test.json', {'utc': now(), 'primary_condition': 'D3',
         'rule': 'All four fixed conditions reported. No test-based choice of features or references.',
         'validation_sha256': digest(a.out / 'val.json'), 'models': {k: digest(a.out / f'{k}.pt') for k in models}})
    assert digest(__file__) == config['code_sha256']
    assert digest(HERE / 'protocol-zh.txt') == config['protocol_sha256']
    score('test', models, d, x, refs, query, groups, y, votes, a.out)
    save(a.out / 'completed.json', {'utc': now(), 'test_sha256': digest(a.out / 'test.json')})
    print('Completed', a.out, flush=True)


if __name__ == '__main__':
    main()
