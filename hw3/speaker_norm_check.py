"""Diagnostic (not part of the proposed design): are the speaker-consistent errors caused by each
speaker's resting voice? Standardise every feature per speaker, using all of that speaker's clips
(no labels; transductive for test speakers, a common SER practice), then retrain the HW2 concat model
with the same recipe and compare neutral clips that listeners clearly heard as neutral.

    python3 hw3/speaker_norm_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
sys.path.insert(0, str(ROOT / 'hw3'))
from train import load_data, speaker_split  # noqa: E402
from train_variants import train_one  # noqa: E402
from analysis import audio_votes  # noqa: E402

E = 'ANG DIS FEA HAP NEU SAD'.split()


def speaker_normalise(X, speakers):
    Xn = X.copy()
    for s in np.unique(speakers):
        m = speakers == s
        flat = X[m].reshape(-1, X.shape[-1])
        Xn[m] = (X[m] - flat.mean(0)) / (flat.std(0) + 1e-6)
    return Xn


def main():
    X, y, meta = load_data()
    Xn = speaker_normalise(X, meta['speaker'])
    test = np.where(np.isin(meta['speaker'], speaker_split(meta['speaker'])['test']))[0]
    yt = y[test]
    votes = audio_votes(meta['fname'][test])
    sup = votes[np.arange(len(yt)), yt] / votes.sum(1)
    base = np.load(ROOT / 'hw3' / 'results' / 'test_predictions.npz')
    out = {}
    for seed in (7, 11, 23):
        model, r = train_one('concat', seed, (Xn, y, meta), tag='_spknorm')
        Xs = (Xn[test] - r_mean(seed)[0]) / r_mean(seed)[1]
        with torch.no_grad():
            pred = model(torch.as_tensor(Xs, dtype=torch.float32)).argmax(1).numpy()
        neu = (yt == 4) & (sup >= 0.6)
        clear = (sup >= 0.6)
        out[seed] = {'test_acc': r['test']['acc'], 'macro_f1': r['test']['macro_f1'],
                     'neutral_heard_neutral_acc': float(np.mean(pred[neu] == 4)),
                     'baseline_neutral_heard_neutral_acc': float(np.mean(base[f'concat_s{seed}'][neu] == 4)),
                     'listener_clear_acc': float(np.mean(pred[clear] == yt[clear])),
                     'baseline_listener_clear_acc': float(np.mean(base[f'concat_s{seed}'][clear] == yt[clear])),
                     'per_class_acc': {E[c]: float(np.mean(pred[yt == c] == c)) for c in range(6)},
                     'baseline_per_class_acc': {E[c]: float(np.mean(base[f'concat_s{seed}'][yt == c] == c)) for c in range(6)}}
        np.save(ROOT / 'hw3' / 'runs' / f'concat_spknorm_s{seed}' / 'test_pred.npy', pred)
    (ROOT / 'hw3' / 'results' / 'speaker_norm_check.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def r_mean(seed):
    ck = torch.load(ROOT / 'hw3' / 'runs' / f'concat_spknorm_s{seed}' / 'model.pt', weights_only=False)
    return ck['mean'], ck['std']


if __name__ == '__main__':
    main()
