"""Re-extract segment features for the TEST clips under volume (gain) perturbations.

Variants (19 in total):
  clean            same pipeline, no change (sanity check against data/features_K8.npz)
  global_-6, global_+6       whole clip quieter / louder by 6 dB
  local_-6_s{j}, local_+6_s{j}   only segment j (j = 0..7) quieter / louder by 6 dB

To keep the comparison clean, the silence trimming and the 8 segment boundaries are taken
from the *clean* clip, so a perturbation cannot move where a segment starts or ends.
YIN's F0 estimate is amplitude-invariant within a frame, so the clean F0 track is reused;
the voicing mask is recomputed because it depends on frame energy. All other frame features
(MFCC, delta-MFCC, RMS, centroid, ZCR) are recomputed from the perturbed waveform. The
waveform is not clipped after +6 dB (floats), so this tests the feature pipeline, not a
real recorder.

    python3 hw3/perturb_features.py     # writes data/hw3_perturbed.npz
"""
import multiprocessing as mp
import sys
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
import features as F  # noqa: E402
from train import load_data, speaker_split  # noqa: E402

GAIN_DB = 6
VARIANTS = (['clean', f'global_-{GAIN_DB}', f'global_+{GAIN_DB}']
            + [f'local_{s}{GAIN_DB}_s{j}' for s in '-+' for j in range(F.K)])


def frame_feats_with_f0(y, f0):
    sr, HOP, N_FFT = F.SR, F.HOP, F.N_FFT
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=F.N_MFCC, n_fft=N_FFT, hop_length=HOP)
    width = min(9, mfcc.shape[1] - (1 - mfcc.shape[1] % 2)) if mfcc.shape[1] >= 3 else 0
    dmfcc = librosa.feature.delta(mfcc, width=width) if width >= 3 else np.zeros_like(mfcc)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)[0] / (sr / 2)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0]
    n = min(mfcc.shape[1], len(rms), len(f0))
    logrms = np.log(rms[:n] + 1e-5)
    voiced = (logrms > np.percentile(logrms, 30)) & (f0[:n] > F.FMIN * 1.05) & (f0[:n] < F.FMAX * 0.95)
    return {'mfcc': mfcc[:, :n].T, 'dmfcc': dmfcc[:, :n].T, 'logf0': np.log(f0[:n]), 'voiced': voiced,
            'logrms': logrms, 'cent': cent[:n], 'zcr': zcr[:n]}


def work(path):
    y = F.load(path)
    yt, _ = F.trim(y)
    f0 = librosa.yin(yt, fmin=F.FMIN, fmax=F.FMAX, sr=F.SR, frame_length=1024, hop_length=F.HOP)
    n = len(frame_feats_with_f0(yt, f0)['logrms'])
    bounds = F.segment_bounds(n)
    out = []
    for v in VARIANTS:
        g = np.ones(len(yt), dtype=np.float32)
        if v.startswith('global'):
            g[:] = 10 ** (int(v.split('_')[1]) / 20)
        elif v.startswith('local'):
            _, db, seg = v.split('_')
            lo, hi = bounds[int(seg[1:])]
            g[lo * F.HOP:hi * F.HOP] = 10 ** (int(db) / 20)
        ff = frame_feats_with_f0(yt * g, f0)
        out.append(np.stack([F.summarise(ff, lo, hi) for lo, hi in bounds]))
    return np.stack(out)


def main():
    X, y, meta = load_data()
    groups = speaker_split(meta['speaker'])
    test = np.where(np.isin(meta['speaker'], groups['test']))[0]
    paths = [str(ROOT / 'data' / 'crema' / f) for f in meta['fname'][test]]
    with mp.Pool(2) as pool:
        P = np.stack(pool.map(work, paths, chunksize=16))       # (N, V, K, D)
    P = P.transpose(1, 0, 2, 3)
    diff = np.abs(P[0] - X[test]).max()
    print('clean re-extraction max abs diff vs stored features:', diff)
    np.savez_compressed(ROOT / 'data' / 'hw3_perturbed.npz', P=P, variants=np.array(VARIANTS),
                        test_idx=test)
    print('saved', P.shape)


if __name__ == '__main__':
    main()
