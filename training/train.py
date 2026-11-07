"""Train the SegmentMLP speech-emotion classifier and a shallow logistic-regression benchmark.

The training loop is written out explicitly so each step of the assignment is visible:
  1. initialise parameters        (nn.Linear default init, fixed seed)
  2. forward propagation          model(xb)
  3. loss                         cross-entropy
  4-5. backpropagation/gradients  loss.backward()
  6. gradient-descent update      SGD with momentum, opt.step()
  7. repeat over epochs; keep the checkpoint with the best validation accuracy

Data are split by SPEAKER (actor), so test speakers are never seen in training.
Speakers 1001 and 1002 (the clips shown in the web app) are always in the test set.

    python3 training/train.py                       # default config, writes model/
    python3 training/train.py --epochs 80 --lr 0.03 --hidden 256 --layers 2 --activation gelu
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

try:
    from .model import SegmentMLP
    from .features import EMOTIONS, FEATURE_NAMES, PROSODY_IDX
except ImportError:  # run as a script
    from model import SegmentMLP
    from features import EMOTIONS, FEATURE_NAMES, PROSODY_IDX

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / 'data' / 'features_K8.npz'
MODEL_DIR = ROOT / 'model'

DEFAULTS = dict(lr=0.02, momentum=0.9, weight_decay=5e-3, epochs=60, batch_size=64,
                hidden=128, latent=64, layers=1, activation='relu', dropout=0.5, seed=7,
                feature_set='all')


# ----------------------------------------------------------------------------- data
def feature_idx(feature_set):
    """'all' = MFCC + prosody (88 dims); 'mfcc' = MFCC/delta statistics only (80 dims),
    used as an ablation to test whether explicit pitch/energy evidence matters."""
    if feature_set == 'mfcc':
        return [i for i in range(len(FEATURE_NAMES)) if i not in PROSODY_IDX]
    return list(range(len(FEATURE_NAMES)))


def load_data(path=FEATURES):
    d = np.load(path)
    y = np.array([EMOTIONS.index(e) for e in d['emotion']])
    meta = {k: d[k] for k in ('fname', 'speaker', 'sentence', 'emotion', 'level')}
    return d['X'].astype(np.float32), y, meta


def speaker_split(speakers, seed=7, n_test=18, n_val=9, fixed_test=('1001', '1002')):
    uniq = sorted(set(speakers.tolist()))
    rest = [s for s in uniq if s not in fixed_test]
    rng = np.random.default_rng(seed)
    rng.shuffle(rest)
    test = list(fixed_test) + rest[:n_test - len(fixed_test)]
    val = rest[n_test - len(fixed_test):n_test - len(fixed_test) + n_val]
    train = [s for s in uniq if s not in test and s not in val]
    return {'train': sorted(train), 'val': sorted(val), 'test': sorted(test)}


def make_splits(X, y, meta, seed=7):
    groups = speaker_split(meta['speaker'], seed=seed)
    idx = {k: np.where(np.isin(meta['speaker'], v))[0] for k, v in groups.items()}
    mean = X[idx['train']].reshape(-1, X.shape[-1]).mean(0)
    std = X[idx['train']].reshape(-1, X.shape[-1]).std(0) + 1e-6
    Xs = (X - mean) / std
    return Xs, idx, groups, mean, std


# ----------------------------------------------------------------------------- training
def evaluate(model, X, y, batch=1024):
    model.eval()
    losses, preds = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            logits = model(X[i:i + batch])
            losses.append(F.cross_entropy(logits, y[i:i + batch], reduction='sum').item())
            preds.append(logits.argmax(1))
    pred = torch.cat(preds)
    return sum(losses) / len(X), (pred == y).float().mean().item(), pred.numpy()


def train(config=None, on_epoch=None, should_stop=None, save=True, out_dir=MODEL_DIR, verbose=True):
    """Train one model. `on_epoch(record)` is called after every epoch (used by the web
    app's live training monitor); `should_stop()` lets the server cancel a run."""
    cfg = {**DEFAULTS, **(config or {})}
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    X, y, meta = load_data()
    fidx = feature_idx(cfg['feature_set'])
    X = X[..., fidx]
    Xs, idx, groups, mean, std = make_splits(X, y, meta, seed=7)   # split fixed across runs
    T = {k: (torch.tensor(Xs[v]), torch.tensor(y[v])) for k, v in idx.items()}

    # 1. initialise parameters
    model = SegmentMLP(X.shape[-1], k=X.shape[1], n_classes=len(EMOTIONS), hidden=cfg['hidden'],
                       latent=cfg['latent'], n_hidden_layers=cfg['layers'],
                       activation=cfg['activation'], dropout=cfg['dropout'])
    opt = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                          weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'])
    history, best = [], {'val_acc': -1}
    Xtr, ytr = T['train']
    t0 = time.time()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        run_loss, run_correct = 0.0, 0
        for i in range(0, len(Xtr), cfg['batch_size']):
            b = perm[i:i + cfg['batch_size']]
            xb, yb = Xtr[b], ytr[b]
            logits = model(xb)                       # 2. forward propagation
            loss = F.cross_entropy(logits, yb)       # 3. loss
            opt.zero_grad()
            loss.backward()                          # 4-5. backpropagation -> parameter gradients
            opt.step()                               # 6. gradient-descent update
            run_loss += loss.item() * len(b)
            run_correct += (logits.argmax(1) == yb).sum().item()
        sched.step()
        val_loss, val_acc, _ = evaluate(model, *T['val'])
        rec = {'epoch': epoch, 'train_loss': run_loss / len(Xtr), 'train_acc': run_correct / len(Xtr),
               'val_loss': val_loss, 'val_acc': val_acc, 'lr': sched.get_last_lr()[0],
               'seconds': round(time.time() - t0, 2)}
        history.append(rec)
        if val_acc > best['val_acc']:
            best = {'val_acc': val_acc, 'epoch': epoch,
                    'state': {k: v.clone() for k, v in model.state_dict().items()}}
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"epoch {epoch:3d}  train {rec['train_loss']:.3f}/{rec['train_acc']:.3f}  "
                  f"val {val_loss:.3f}/{val_acc:.3f}", flush=True)
        if on_epoch:
            on_epoch(rec)
        if should_stop and should_stop():
            break

    # 7. keep the best-validation checkpoint, then evaluate once on the held-out test speakers
    model.load_state_dict(best['state'])
    results = {'config': cfg, 'best_epoch': best['epoch'], 'history': history,
               'splits': {k: len(v) for k, v in idx.items()}, 'speakers': groups}
    for split in ('train', 'val', 'test'):
        loss, acc, pred = evaluate(model, *T[split])
        results[split] = {'loss': loss, 'acc': acc,
                          'macro_f1': f1_score(T[split][1].numpy(), pred, average='macro')}
        if split == 'test':
            results['test']['confusion'] = confusion_matrix(T[split][1].numpy(), pred,
                                                            labels=range(len(EMOTIONS))).tolist()
    model.eval()
    with torch.no_grad():   # mean latent slot over training clips: the 'neutral' replacement for probes
        results_zbar = model.encode(Xtr).mean(0).numpy()
    cache = ROOT / 'data' / f"baselines_{cfg['feature_set']}.json"   # fixed split -> reusable
    if cache.exists():
        results['baselines'] = json.loads(cache.read_text())
    else:
        results['baselines'] = shallow_baselines(Xs, y, idx)
        cache.write_text(json.dumps(results['baselines']))
    results['classes'] = EMOTIONS
    if save:
        save_model(model, cfg, mean, std, results, out_dir, fidx, results_zbar)
    return model, results


def shallow_baselines(Xs, y, idx):
    """Shallow benchmarks on the same speaker split: multinomial logistic regression
    on (a) utterance-mean features and (b) all 8 segments concatenated."""
    out = {'chance': 1 / len(EMOTIONS),
           'majority': float(np.bincount(y[idx['train']]).max() / len(idx['train']))}
    for name, f in [('logreg_mean', lambda a: a.mean(1)), ('logreg_concat', lambda a: a.reshape(len(a), -1))]:
        # choose the regularisation strength C on the validation speakers (fair tuning)
        fits = []
        for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            c = LogisticRegression(max_iter=3000, C=C).fit(f(Xs[idx['train']]), y[idx['train']])
            fits.append((c.score(f(Xs[idx['val']]), y[idx['val']]), C, c))
        _, C, clf = max(fits, key=lambda t: t[0])
        pred = clf.predict(f(Xs[idx['test']]))
        out[name] = {'test_acc': float((pred == y[idx['test']]).mean()),
                     'test_macro_f1': float(f1_score(y[idx['test']], pred, average='macro')),
                     'val_acc': float(clf.score(f(Xs[idx['val']]), y[idx['val']])), 'C': C}
    return out


def save_model(model, cfg, mean, std, results, out_dir=MODEL_DIR, fidx=None, zbar=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'config': cfg, 'mean': mean, 'std': std,
                'classes': EMOTIONS, 'feature_names': FEATURE_NAMES,
                'feature_idx': fidx if fidx is not None else list(range(len(FEATURE_NAMES))),
                'train_mean_slots': zbar,
                'arch': dict(d_in=model.encoder[0].in_features, k=model.k, n_classes=model.n_classes)},
               out_dir / 'segment_mlp.pt')
    (out_dir / 'metrics.json').write_text(json.dumps(results, indent=1))
    try:
        plot_curves(results, out_dir / 'curves.png')
    except Exception as exc:  # plotting is optional
        print('plot skipped:', exc)


def plot_curves(results, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h = results['history']
    ep = [r['epoch'] for r in h]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ep, [r['train_loss'] for r in h], label='train')
    ax[0].plot(ep, [r['val_loss'] for r in h], label='validation')
    ax[0].set(title='Cross-entropy loss', xlabel='epoch')
    ax[1].plot(ep, [r['train_acc'] for r in h], label='train')
    ax[1].plot(ep, [r['val_acc'] for r in h], label='validation')
    b = results['baselines']['logreg_concat']['val_acc']
    ax[1].axhline(b, ls='--', color='gray', label='log. regression (val)')
    ax[1].set(title='Accuracy', xlabel='epoch')
    for a in ax:
        a.legend(frameon=False)
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main():
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        p.add_argument('--' + k, type=type(v), default=v)
    p.add_argument('--out', default=str(MODEL_DIR))
    args = vars(p.parse_args())
    out = args.pop('out')
    _, r = train(args, out_dir=out)
    b = r['baselines']
    print(f"\nTEST  SegmentMLP acc {r['test']['acc']:.3f}  macro-F1 {r['test']['macro_f1']:.3f}")
    print(f"TEST  LogReg(mean)   acc {b['logreg_mean']['test_acc']:.3f}")
    print(f"TEST  LogReg(concat) acc {b['logreg_concat']['test_acc']:.3f}   chance {b['chance']:.3f}")


if __name__ == '__main__':
    main()
