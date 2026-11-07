"""HW3 variants of SegmentMLP that change how the 8 segment slots are combined.

All variants share the HW2 encoder architecture: a shared MLP maps each segment
to a latent slot. concat_rel also changes the two input columns before encoding.

  concat      HW2 baseline. logits = W [z_1; ...; z_8] + b.
              Each time position has its own read-out weights.
  mean        Ablation. h = mean_k z_k, logits = W h + b.
              Removes the position-specific read-out but learns no weighting.
  attn        Attention pooling. s_k = v^T tanh(A z_k), alpha = softmax(s), h = sum_k alpha_k z_k.
              The score is computed from the same representation that is pooled.
  attn_rel    A separate scorer receives all 88 features, with two amplitude-related columns
              (log-RMS mean, MFCC-0 mean) centred before scoring. This approximately stabilises
              scores under the tested global gains; it is not exact numerical invariance.
              The value path (z_k) still receives uncentred inputs.

Every pooled variant exposes encode(), weights() and pool_and_classify(), so the analysis can
swap a clean representation with a perturbed one while holding the weights fixed, and vice versa.
"""
import torch
from torch import nn

ACT = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}
POOLINGS = ('concat', 'mean', 'attn', 'attn_rel', 'concat_rel', 'attn_pos')
# attn_pos    Follow-up: 'attn' plus a learned position embedding p_k added to each latent slot
#             (z_k <- z_k + p_k). Probes position information, but also changes the parameterisation;
#             it does not uniquely identify why the plain attention model lost accuracy.
# concat_rel  Follow-up experiment, added after the 2x2 path swap: HW2's concat read-out, but the
#             absolute-level columns are centred on the utterance mean *before the encoder* (value
#             path). Tests whether gain sensitivity lives in the representation rather than the pooling.


class PooledSegmentModel(nn.Module):
    def __init__(self, d_in, k=8, n_classes=6, hidden=128, latent=64, pooling='concat',
                 dropout=0.5, score_hidden=32, level_idx=(), activation='relu'):
        super().__init__()
        assert pooling in POOLINGS
        self.k, self.latent, self.n_classes, self.pooling = k, latent, n_classes, pooling
        act = ACT[activation]
        # identical to HW2's encoder (1 hidden layer)
        self.encoder = nn.Sequential(nn.Linear(d_in, hidden), act(), nn.Dropout(dropout),
                                     nn.Linear(hidden, latent), act())
        self.drop = nn.Dropout(dropout)
        self.rel_input = pooling == 'concat_rel'
        if self.rel_input:
            pooling = self.pooling = 'concat'
        self.use_pos = pooling == 'attn_pos'
        if self.use_pos:
            pooling = self.pooling = 'attn'
            self.pos = nn.Parameter(torch.zeros(k, latent))
        if pooling == 'concat':
            self.head = nn.Linear(k * latent, n_classes)
        else:
            self.head = nn.Linear(latent, n_classes)
        if pooling == 'attn':
            self.score = nn.Sequential(nn.Linear(latent, score_hidden), nn.Tanh(),
                                       nn.Linear(score_hidden, 1, bias=False))
        if pooling == 'attn_rel' or self.rel_input:
            self.register_buffer('level_idx', torch.as_tensor(list(level_idx), dtype=torch.long))
        if pooling == 'attn_rel':
            self.score = nn.Sequential(nn.Linear(d_in, score_hidden), nn.Tanh(),
                                       nn.Linear(score_hidden, 1, bias=False))

    # ------------------------------------------------------------------ pieces
    def encode(self, x):
        z = self.encoder(self.relative_view(x) if self.rel_input else x)
        return z + self.pos if self.use_pos else z

    def relative_view(self, x):
        """Centre the absolute-level columns on the utterance mean (over segments)."""
        x = x.clone()
        cols = x[..., self.level_idx]
        x[..., self.level_idx] = cols - cols.mean(1, keepdim=True)
        return x

    def weights(self, x, z=None):
        """Pooling weights alpha (B, K). Uniform for 'mean'; None for 'concat'."""
        if self.pooling == 'concat':
            return None
        if self.pooling == 'mean':
            return torch.full(x.shape[:2], 1.0 / self.k, device=x.device)
        if self.pooling == 'attn':
            z = self.encode(x) if z is None else z
            s = self.score(z).squeeze(-1)
        else:
            s = self.score(self.relative_view(x)).squeeze(-1)
        return torch.softmax(s, dim=1)

    def pool_and_classify(self, z, alpha):
        if self.pooling == 'concat':
            return self.head(self.drop(z).flatten(1))
        h = (alpha.unsqueeze(-1) * z).sum(1)
        return self.head(self.drop(h))

    def forward(self, x, return_parts=False):
        z = self.encode(x)
        a = self.weights(x, z)
        logits = self.pool_and_classify(z, a)
        return (logits, z, a) if return_parts else logits

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
