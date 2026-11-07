"""Segment-level acoustic features for speech emotion recognition.

Each utterance is trimmed of leading/trailing silence and split into K equal-duration
segments ("slots"). For every segment we summarise frame-level features:

  * 20 MFCCs + 20 delta-MFCCs          -> mean and std  (80 dims)
  * log F0 (YIN, voiced frames only)    -> mean, std, range, voiced fraction (4 dims)
  * log RMS energy                      -> mean, std (2 dims)
  * spectral centroid, zero-crossing    -> mean (2 dims)

giving an array of shape (K, 88). MFCCs alone carry little pitch information, so the
F0 and energy statistics are included explicitly: emotional prosody is largely carried
by pitch and loudness contours.

Run as a script to extract features for the whole corpus:
    python3 training/features.py            # writes data/features_K8.npz
"""
import argparse
import multiprocessing as mp
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
K = 8
N_MFCC = 20
HOP = 160          # 10 ms
N_FFT = 512
FMIN, FMAX = 65.0, 500.0
EMOTIONS = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
EMOTION_NAMES = {'ANG': 'Anger', 'DIS': 'Disgust', 'FEA': 'Fear', 'HAP': 'Happiness', 'NEU': 'Neutral', 'SAD': 'Sadness'}

FEATURE_NAMES = (
    [f'mfcc{i}_mean' for i in range(N_MFCC)] + [f'dmfcc{i}_mean' for i in range(N_MFCC)]
    + [f'mfcc{i}_std' for i in range(N_MFCC)] + [f'dmfcc{i}_std' for i in range(N_MFCC)]
    + ['logf0_mean', 'logf0_std', 'logf0_range', 'voiced_frac', 'logrms_mean', 'logrms_std',
       'centroid_mean', 'zcr_mean']
)
D = len(FEATURE_NAMES)  # 88
PROSODY_IDX = list(range(4 * N_MFCC, D))  # indices of F0/energy/spectral summaries


def load(path_or_bytes, sr=SR):
    """Load mono audio at SR. Accepts a path or a file-like object."""
    y, _ = librosa.load(path_or_bytes, sr=sr, mono=True)
    return y.astype(np.float32)


def trim(y):
    yt, idx = librosa.effects.trim(y, top_db=30)
    return (yt if len(yt) > SR // 10 else y), idx


def frame_features(y, sr=SR):
    """Frame-level feature matrix, shape (frames, channels), plus a voiced mask."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    width = min(9, mfcc.shape[1] - (1 - mfcc.shape[1] % 2)) if mfcc.shape[1] >= 3 else 0
    dmfcc = librosa.feature.delta(mfcc, width=width) if width >= 3 else np.zeros_like(mfcc)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)[0] / (sr / 2)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0]
    f0 = librosa.yin(y, fmin=FMIN, fmax=FMAX, sr=sr, frame_length=1024, hop_length=HOP)
    n = min(mfcc.shape[1], len(rms), len(f0))
    logrms = np.log(rms[:n] + 1e-5)
    # Voicing heuristic: energetic frames with an in-range F0 estimate.
    voiced = (logrms > np.percentile(logrms, 30)) & (f0[:n] > FMIN * 1.05) & (f0[:n] < FMAX * 0.95)
    return {
        'mfcc': mfcc[:, :n].T, 'dmfcc': dmfcc[:, :n].T, 'logf0': np.log(f0[:n]), 'voiced': voiced,
        'logrms': logrms, 'cent': cent[:n], 'zcr': zcr[:n],
    }


def summarise(ff, lo, hi):
    """Summarise frames [lo, hi) into one D-dim vector."""
    sl = slice(lo, max(hi, lo + 1))
    m, d = ff['mfcc'][sl], ff['dmfcc'][sl]
    v = ff['voiced'][sl]
    f0 = ff['logf0'][sl][v]
    if len(f0) >= 2:
        f0_stats = [f0.mean(), f0.std(), f0.max() - f0.min(), v.mean()]
    elif len(f0) == 1:
        f0_stats = [f0[0], 0.0, 0.0, v.mean()]
    else:  # unvoiced segment: fall back to the utterance-level voiced mean
        allf0 = ff['logf0'][ff['voiced']]
        f0_stats = [allf0.mean() if len(allf0) else np.log(150.0), 0.0, 0.0, 0.0]
    r = ff['logrms'][sl]
    return np.concatenate([
        m.mean(0), d.mean(0), m.std(0), d.std(0),
        f0_stats, [r.mean(), r.std(), ff['cent'][sl].mean(), ff['zcr'][sl].mean()],
    ]).astype(np.float32)


def segment_bounds(n_frames, k=K):
    edges = np.linspace(0, n_frames, k + 1).round().astype(int)
    return [(edges[i], edges[i + 1]) for i in range(k)]


def extract(y, sr=SR, k=K, return_bounds=False):
    """Audio -> (k, D) segment features. Optionally return segment times (seconds, untrimmed)."""
    yt, (start, _) = trim(y)
    ff = frame_features(yt, sr)
    n = len(ff['logrms'])
    bounds = segment_bounds(n, k)
    X = np.stack([summarise(ff, lo, hi) for lo, hi in bounds])
    if return_bounds:
        t0 = start / sr if len(yt) != len(y) else 0.0
        times = [(t0 + lo * HOP / sr, t0 + hi * HOP / sr) for lo, hi in bounds]
        return X, times
    return X


def parse_name(name):
    speaker, sentence, emotion, level = Path(name).stem.split('_')
    return {'fname': Path(name).name, 'speaker': speaker, 'sentence': sentence, 'emotion': emotion, 'level': level}


def _work(path):
    try:
        return path.name, extract(load(path))
    except Exception as exc:  # a handful of CREMA-D clips are very short or near-silent
        return path.name, exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', default=str(ROOT / 'data' / 'crema'))
    parser.add_argument('--out', default=str(ROOT / 'data' / f'features_K{K}.npz'))
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count()))
    args = parser.parse_args()
    paths = sorted(Path(args.audio).glob('*.wav'))
    print(f'{len(paths)} files, {args.workers} workers')
    X, meta, bad = [], [], []
    with mp.Pool(args.workers) as pool:
        for i, (name, res) in enumerate(pool.imap(_work, paths, chunksize=16), 1):
            if isinstance(res, Exception) or not np.all(np.isfinite(res)):
                bad.append(name)
            else:
                X.append(res)
                meta.append(parse_name(name))
            if i % 500 == 0:
                print(f'{i}/{len(paths)}', flush=True)
    X = np.stack(X)
    cols = {key: np.array([m[key] for m in meta]) for key in meta[0]}
    np.savez_compressed(args.out, X=X, feature_names=np.array(FEATURE_NAMES), **cols)
    print(f'saved {X.shape} to {args.out}; skipped {len(bad)}: {bad[:10]}')


if __name__ == '__main__':
    main()
