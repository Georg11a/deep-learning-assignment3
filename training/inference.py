"""Load a trained SegmentMLP and run predictions with per-segment read-outs.

Shared by the web server (server.py) and the pilot analysis (probe.py).
"""
from pathlib import Path

import numpy as np
import torch

try:
    from . import features as F
    from .model import SegmentMLP
except ImportError:
    import features as F
    from model import SegmentMLP

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / 'model' / 'segment_mlp.pt'


class Predictor:
    def __init__(self, ckpt=DEFAULT_CKPT):
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        cfg, arch = ck['config'], ck['arch']
        self.model = SegmentMLP(arch['d_in'], k=arch['k'], n_classes=arch['n_classes'], hidden=cfg['hidden'],
                                latent=cfg['latent'], n_hidden_layers=cfg['layers'],
                                activation=cfg['activation'], dropout=cfg['dropout'])
        self.model.load_state_dict(ck['state_dict'])
        self.model.eval()
        self.mean, self.std = ck['mean'], ck['std']
        self.fidx = ck.get('feature_idx', list(range(F.D)))
        self.classes = ck['classes']
        self.config = cfg
        zbar = ck.get('train_mean_slots')
        self.train_mean_slots = torch.as_tensor(zbar) if zbar is not None else torch.zeros(arch['k'], cfg['latent'])

    def standardise(self, X):
        """Raw (N, K, 88) features -> standardised model inputs (N, K, d_in)."""
        return ((X[..., self.fidx] - self.mean) / self.std).astype(np.float32)

    @torch.no_grad()
    def latents(self, Xs):
        return self.model.encode(torch.as_tensor(Xs))

    @torch.no_grad()
    def from_latents(self, z):
        """Latent slots (N, K, H) -> (log-probabilities (N, C), readouts (N, K, C))."""
        logits = self.model.classify(z)
        return torch.log_softmax(logits, -1), self.model.readout(z)

    def predict_features(self, X):
        z = self.latents(self.standardise(X))
        logp, r = self.from_latents(z)
        return logp.exp().numpy(), r.numpy(), z.numpy()

    def predict_audio(self, y):
        X, times = F.extract(y, return_bounds=True)
        probs, r, _ = self.predict_features(X[None])
        return self.describe(probs[0], r[0], times)

    def describe(self, probs, readout, times=None):
        """JSON-friendly result. `contrast[t]` is the positive part of segment t's contribution
        to the predicted class minus its mean contribution to the other classes (ReaLVR's
        answer-contrast credit, adapted to a linear read-out)."""
        top = int(np.argmax(probs))
        others = [c for c in range(len(self.classes)) if c != top]
        contrast = np.maximum(readout[:, top] - readout[:, others].mean(1), 0)
        out = {
            'classes': [F.EMOTION_NAMES[c] for c in self.classes],
            'codes': list(self.classes),
            'probs': [round(float(p), 4) for p in probs],
            'top': F.EMOTION_NAMES[self.classes[top]],
            'segments': [{'contribution': [round(float(v), 4) for v in readout[t]],
                          'contrast': round(float(contrast[t]), 4)} for t in range(len(readout))],
        }
        if times is not None:
            for seg, (a, b) in zip(out['segments'], times):
                seg['start'], seg['end'] = round(a, 3), round(b, 3)
        return out
