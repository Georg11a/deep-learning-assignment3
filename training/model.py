"""SegmentMLP: a small multi-layer perceptron applied to K time segments of an utterance.

    x  : (B, K, D)   segment features (D = 88 acoustic summaries per segment)
    z_t = f(x_t)     a shared MLP encoder maps every segment to a latent slot z_t (B, K, H)
    logits = W [z_1; ...; z_K] + b      one linear read-out over the concatenated slots

Because the read-out is linear in the concatenation, the class logit decomposes
exactly into per-segment contributions:

    logit_y = b_y + sum_t r_t(y),   r_t(y) = W[y, slice_t] . z_t

r_t(y) is the analogue of ReaLVR's "answer readout" of latent token t, and lets us
ask which segments the decision actually reads (see training/probe.py).
"""
import torch
from torch import nn

ACTIVATIONS = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU, 'sigmoid': nn.Sigmoid}


class SegmentMLP(nn.Module):
    def __init__(self, d_in, k=8, n_classes=6, hidden=128, latent=64, n_hidden_layers=1,
                 activation='relu', dropout=0.3):
        super().__init__()
        self.k, self.latent, self.n_classes = k, latent, n_classes
        act = ACTIVATIONS[activation]
        layers, width = [], d_in
        for _ in range(n_hidden_layers):          # hidden layers of the shared encoder
            layers += [nn.Linear(width, hidden), act(), nn.Dropout(dropout)]
            width = hidden
        layers += [nn.Linear(width, latent), act()]   # latent slot z_t
        self.encoder = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(k * latent, n_classes)

    def encode(self, x):
        """(B, K, D) -> latent slots (B, K, latent)."""
        return self.encoder(x)

    def classify(self, z):
        """Latent slots (B, K, latent) -> logits (B, n_classes)."""
        return self.head(self.drop(z).flatten(1))

    def forward(self, x):
        return self.classify(self.encode(x))

    def readout(self, z):
        """Per-segment contribution to every class logit: (B, K, n_classes).
        Sums over K (plus the bias) to the logits in eval mode."""
        w = self.head.weight.view(self.n_classes, self.k, self.latent)   # (C, K, H)
        return torch.einsum('bkh,ckh->bkc', z, w)
