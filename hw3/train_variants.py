"""Train the HW3 pooling variants under the HW2 recipe.

Same data, speaker split, standardisation, optimiser (SGD + momentum 0.9, lr 0.02,
weight decay 5e-3, cosine schedule), 60 epochs, batch 64, dropout 0.5, and best-validation
checkpoint as HW2. Variants change pooling, position encoding or input centering.
Each variant runs with 3 seeds. Existing run folders are never overwritten.

    python3 hw3/train_variants.py                         # all variants x seeds 7, 11, 23
    python3 hw3/train_variants.py --poolings attn --seeds 7
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
sys.path.insert(0, str(ROOT / 'hw3'))
from features import EMOTIONS, FEATURE_NAMES  # noqa: E402
from train import load_data, make_splits  # noqa: E402
from models import PooledSegmentModel, POOLINGS  # noqa: E402

RUNS = ROOT / 'hw3' / 'runs'
CFG = dict(lr=0.02, momentum=0.9, weight_decay=5e-3, epochs=60, batch_size=64,
           hidden=128, latent=64, dropout=0.5)
LEVEL_FEATURES = ('mfcc0_mean', 'logrms_mean')
LEVEL_IDX = [FEATURE_NAMES.index(n) for n in LEVEL_FEATURES]


def build(pooling, d_in, k):
    return PooledSegmentModel(d_in, k=k, n_classes=len(EMOTIONS), hidden=CFG['hidden'],
                              latent=CFG['latent'], pooling=pooling, dropout=CFG['dropout'],
                              level_idx=LEVEL_IDX)


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    logits = model(X)
    pred = logits.argmax(1)
    return F.cross_entropy(logits, y).item(), (pred == y).float().mean().item(), pred.numpy()


def train_one(pooling, seed, data=None, verbose=True, tag=''):
    out = RUNS / f'{pooling}{tag}_s{seed}'
    if out.exists():
        raise FileExistsError(f'Preserving {out}; use --runs-dir for a new reproduction.')
    X, y, meta = data or load_data()
    Xs, idx, groups, mean, std = make_splits(X, y, meta, seed=7)      # split fixed across runs
    T = {k: (torch.tensor(Xs[v]), torch.tensor(y[v])) for k, v in idx.items()}
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build(pooling, X.shape[-1], X.shape[1])
    opt = torch.optim.SGD(model.parameters(), lr=CFG['lr'], momentum=CFG['momentum'],
                          weight_decay=CFG['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['epochs'])
    Xtr, ytr = T['train']
    history, best, t0 = [], {'val_acc': -1}, time.time()
    for epoch in range(1, CFG['epochs'] + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot, correct, gnorm = 0.0, 0, []
        for i in range(0, len(Xtr), CFG['batch_size']):
            b = perm[i:i + CFG['batch_size']]
            logits = model(Xtr[b])
            loss = F.cross_entropy(logits, ytr[b])
            opt.zero_grad()
            loss.backward()
            gnorm.append(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item())  # measure only
            opt.step()
            tot += loss.item() * len(b)
            correct += (logits.argmax(1) == ytr[b]).sum().item()
        sched.step()
        vl, va, _ = evaluate(model, *T['val'])
        rec = {'epoch': epoch, 'train_loss': tot / len(Xtr), 'train_acc': correct / len(Xtr),
               'val_loss': vl, 'val_acc': va, 'grad_norm': float(np.mean(gnorm)),
               'seconds': round(time.time() - t0, 2)}
        if model.pooling in ('attn', 'attn_rel'):
            model.eval()
            with torch.no_grad():
                a = model.weights(T['val'][0])
            ent = -(a * a.clamp_min(1e-12).log()).sum(1).mean().item()
            rec['val_attn_entropy'] = ent            # log(8) = 2.079 means uniform
            rec['val_attn_max'] = a.max(1).values.mean().item()
        history.append(rec)
        if va > best['val_acc']:
            best = {'val_acc': va, 'epoch': epoch,
                    'state': {k: v.clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best['state'])
    res = {'pooling': pooling, 'seed': seed, 'config': CFG, 'best_epoch': best['epoch'],
           'n_params': model.n_params(), 'train_seconds': history[-1]['seconds'], 'history': history}
    for split in ('train', 'val', 'test'):
        loss, acc, pred = evaluate(model, *T[split])
        res[split] = {'loss': loss, 'acc': acc,
                      'macro_f1': float(f1_score(T[split][1].numpy(), pred, average='macro'))}
    res['test']['confusion'] = confusion_matrix(T['test'][1].numpy(), pred,
                                                labels=range(len(EMOTIONS))).tolist()
    out.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'pooling': pooling, 'mean': mean, 'std': std,
                'd_in': X.shape[-1], 'k': X.shape[1], 'level_idx': LEVEL_IDX}, out / 'model.pt')
    (out / 'metrics.json').write_text(json.dumps(res, indent=1))
    if verbose:
        print(f"{pooling + tag:9s} seed {seed:2d}  best ep {best['epoch']:2d}  train {res['train']['acc']:.3f}  "
              f"val {res['val']['acc']:.3f}  test {res['test']['acc']:.3f}  F1 {res['test']['macro_f1']:.3f}  "
              f"params {res['n_params']}", flush=True)
    return model, res


def load_model(pooling, seed, tag=''):
    ck = torch.load(RUNS / f'{pooling}{tag}_s{seed}' / 'model.pt', map_location='cpu', weights_only=False)
    m = build(ck['pooling'], ck['d_in'], ck['k'])
    m.load_state_dict(ck['state_dict'])
    m.eval()
    return m, ck['mean'], ck['std']


def main():
    global RUNS
    p = argparse.ArgumentParser()
    p.add_argument('--poolings', nargs='+', default=list(POOLINGS))
    p.add_argument('--seeds', nargs='+', type=int, default=[7, 11, 23])
    p.add_argument('--runs-dir', type=Path, default=RUNS)
    a = p.parse_args()
    RUNS = a.runs_dir
    data = load_data()
    for pooling in a.poolings:
        for seed in a.seeds:
            train_one(pooling, seed, data)


if __name__ == '__main__':
    main()
