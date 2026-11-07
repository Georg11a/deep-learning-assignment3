"""Figures for the HW3 report and video. Reads hw3/results/*.json and hw3/runs/*/metrics.json.

    python3 hw3/figures.py      # writes hw3/figures/*.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from train import load_data  # noqa: E402

FIG = ROOT / 'hw3' / 'figures'
RES = ROOT / 'hw3' / 'results'
RUNS = ROOT / 'hw3' / 'runs'
SEEDS = (7, 11, 23)
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
COL = {'concat': '#2a78d6', 'mean': '#eb6834', 'attn': '#1baf7a', 'attn_rel': '#eda100',
       'attn_pos': '#e87ba4', 'concat_rel': '#008300', 'concat_spknorm': '#6250d6'}
LABEL = {'concat': 'HW2 baseline\n(concat)', 'mean': 'mean pool\n(ablation)', 'attn': 'attention\npool',
         'attn_rel': 'attention,\nrelative scorer', 'attn_pos': 'attention\n+ position',
         'concat_rel': 'baseline,\nlevel-centred input', 'concat_spknorm': 'baseline,\nspeaker-normalised*'}
E = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'axes.titlesize': 10, 'axes.titleweight': 'bold',
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True})


def tidy(ax):
    ax.spines[['top', 'right']].set_visible(False)


def stat(M, p, f):
    v = [f(M[f'{p}_s{s}']) for s in SEEDS]
    return np.mean(v), np.std(v)


def fig_results(M, spk):
    order = ['concat', 'mean', 'attn', 'attn_rel', 'attn_pos', 'concat_rel']
    panels = [('Test accuracy (%)', lambda m: 100 * m['test']['acc']),
              ('Predictions changed by\nwhole-clip −6 dB (%)', lambda m: 100 * m['robustness']['global_-6']['prediction_change']),
              ('Predictions changed by\none segment ±6 dB (%)', lambda m: 100 * m['robustness']['local_mean']['prediction_change']),
              ('Same speaker & emotion, other\nsentence → answer flips (%)', lambda m: 100 * m['paired']['false_flip_U'])]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.8))
    for ax, (title, f) in zip(axs, panels):
        mu = [stat(M, p, f) for p in order]
        x = np.arange(len(order))
        ax.bar(x, [m for m, _ in mu], color=[COL[p] for p in order], width=0.72, edgecolor=SURF, linewidth=2)
        ax.errorbar(x, [m for m, _ in mu], [s for _, s in mu], fmt='none', ecolor=INK2, capsize=2, lw=1)
        for i, (m, _) in enumerate(mu):
            ax.text(i, m + 0.6, f'{m:.1f}', ha='center', va='bottom', fontsize=7.5, color=INK)
        short = {'concat': 'baseline', 'mean': 'mean pool', 'attn': 'attention', 'attn_rel': 'attn, rel.\nscorer',
                 'attn_pos': 'attn +\nposition', 'concat_rel': 'baseline,\nlevel-centred'}
        ax.set_xticks(x, [short[p] for p in order], fontsize=7, rotation=35, ha='right')
        ax.set_title(title, fontsize=9, pad=12)
        ax.set_ylim(top=max(m for m, _ in mu) * 1.12 if title[0] != 'T' else 58)
        ax.grid(axis='x', visible=False)
        tidy(ax)
    axs[0].set_ylim(45, 58)
    fig.text(0.01, 0.01, 'Mean ± SD over 3 seeds, 1,475 clips from 18 held-out speakers. Lower is better in panels 2–4.',
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / 'fig2_results.png', dpi=160)


def fig_curves():
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
    for p in ['concat', 'mean', 'attn', 'attn_rel', 'attn_pos']:
        h = json.loads((RUNS / f'{p}_s7' / 'metrics.json').read_text())['history']
        ep = [r['epoch'] for r in h]
        axs[0].plot(ep, [r['train_loss'] for r in h], color=COL[p], lw=1.6, label=LABEL[p].replace('\n', ' '))
        axs[1].plot(ep, [100 * r['val_acc'] for r in h], color=COL[p], lw=1.6)
        if 'val_attn_entropy' in h[0]:
            axs[2].plot(ep, [r['val_attn_entropy'] for r in h], color=COL[p], lw=1.6, label=LABEL[p].replace('\n', ' '))
    axs[2].axhline(np.log(8), color=INK2, ls=':', lw=1)
    axs[2].text(60, np.log(8) - 0.01, 'uniform (log 8)', ha='right', va='top', fontsize=7.5, color=INK2)
    axs[0].set(title='Training loss', xlabel='epoch')
    axs[1].set(title='Validation accuracy (%)', xlabel='epoch')
    axs[2].set(title='Attention entropy (validation)', xlabel='epoch')
    axs[0].legend(frameon=False, fontsize=7)
    for a in axs:
        tidy(a)
    fig.text(0.01, 0.01, 'Seed 7. Same recipe for every model: SGD + momentum, lr 0.02 cosine, 60 epochs, best-validation checkpoint.',
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / 'fig3_curves.png', dpi=160)


def fig_path_swap(M):
    cats = [('z_only', 'representation alone flips it'), ('both_alone', 'either alone flips it'),
            ('alpha_only', 'attention weights alone flip it'), ('neither_alone', 'only both together flip it')]
    colors = ['#2a78d6', '#a9c7ee', '#eda100', '#b8b6ae']
    rows = []
    for p in ('attn', 'attn_rel'):
        # global: aggregate over both directions and seeds
        g = {c: 0 for c, _ in cats}
        n = 0
        for s in SEEDS:
            for v in ('global_-6', 'global_+6'):
                r = M[f'{p}_s{s}']['path_swap'][v]
                k = r['n_flip']
                zo, ao = r['z_only_sufficient'], r['alpha_only_sufficient']
                n += k
                # exact counts are kept for local; for global reconstruct from rates (z,a sufficient are not exclusive)
                g['_z'] = g.get('_z', 0) + zo * k
                g['_a'] = g.get('_a', 0) + ao * k
        rows.append((f'{LABEL[p].replace(chr(10), " ")}\nwhole-clip ±6 dB', None, g['_z'] / n, g['_a'] / n, n))
        A = [M[f'{p}_s{s}']['path_swap']['local_aggregate'] for s in SEEDS]
        nl = sum(a['n_flip'] for a in A)
        rows.append((f'{LABEL[p].replace(chr(10), " ")}\none segment ±6 dB', {c: sum(a[c] for a in A) / nl for c, _ in cats},
                     None, None, nl))
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    for i, (lab, shares, zs, as_, n) in enumerate(rows):
        y = len(rows) - 1 - i
        if shares is None:   # global: show the two sufficiency rates as separate thin bars
            ax.barh(y + 0.17, 100 * zs, height=0.3, color=colors[0])
            ax.barh(y - 0.17, 100 * as_, height=0.3, color=colors[2])
            ax.text(100 * zs + 1, y + 0.17, f'representation alone: {100 * zs:.0f}%', va='center', fontsize=7.5)
            ax.text(100 * as_ + 1, y - 0.17, f'attention alone: {100 * as_:.0f}%', va='center', fontsize=7.5)
        else:
            left = 0
            for (c, _), col in zip(cats, colors):
                w = 100 * shares[c]
                ax.barh(y, w, left=left, height=0.55, color=col, edgecolor=SURF, linewidth=2)
                if w > 6:
                    ax.text(left + w / 2, y, f'{w:.0f}%', ha='center', va='center', fontsize=7.5)
                left += w
        ax.text(110, y, f'n = {n}', va='center', ha='right', fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(rows))[::-1], [r[0] for r in rows], fontsize=7.5)
    ax.set_xlim(0, 125)
    ax.set_xlabel('% of clips that the perturbation turned from correct to wrong')
    ax.set_title('2×2 swap: which path is enough to produce the error?')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, [t for _, t in cats], frameon=False, fontsize=7, ncol=4, loc='upper center',
              bbox_to_anchor=(0.45, -0.22))
    ax.grid(axis='y', visible=False)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(FIG / 'fig4_path_swap.png', dpi=160)


def fig_attention(M):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={'width_ratios': [1.3, 1]})
    pos = np.arange(1, 9)
    for p in ('attn', 'attn_rel'):
        w = np.mean([M[f'{p}_s{s}']['attention']['mean_weight_by_position'] for s in SEEDS], 0)
        axs[0].plot(pos, w, '-o', color=COL[p], lw=1.8, ms=5, label=LABEL[p].replace('\n', ' '))
    wn = np.mean([M[f'concat_s{s}']['readout_by_position']['weight_norm'] for s in SEEDS], 0)
    axs[0].plot(pos, wn / wn.sum(), '--s', color=COL['concat'], lw=1.6, ms=5, label='baseline read-out weight norm (scaled)')
    axs[0].axhline(1 / 8, color=INK2, ls=':', lw=1)
    axs[0].set(title='Where the model puts weight', xlabel='segment (1 = start, 8 = end)', ylabel='mean weight')
    axs[0].legend(frameon=False, fontsize=7)
    keys = [('spearman_alpha_vs_logrms', 'loudness'), ('spearman_alpha_vs_voiced_frac', 'voiced fraction'),
            ('spearman_alpha_vs_logf0', 'mean pitch')]
    for j, p in enumerate(('attn', 'attn_rel')):
        v = [np.mean([M[f'{p}_s{s}']['attention'][k] for s in SEEDS]) for k, _ in keys]
        axs[1].barh(np.arange(3) + (0.2 if j == 0 else -0.2), v, height=0.36, color=COL[p], edgecolor=SURF, linewidth=2)
        for i, x in enumerate(v):
            axs[1].text(x + 0.01, i + (0.2 if j == 0 else -0.2), f'{x:.2f}', va='center', fontsize=7.5)
    axs[1].set_yticks(range(3), [t for _, t in keys])
    axs[1].legend([plt.Rectangle((0, 0), 1, 1, color=COL[p]) for p in ('attn', 'attn_rel')],
                  ['attention pool', 'attention, relative scorer'], frameon=False, fontsize=7, loc='lower right')
    axs[1].set_xlim(0, 0.9)
    axs[1].set(title='Rank correlation within each clip:\nattention weight vs segment feature', xlabel='Spearman ρ')
    axs[1].grid(axis='y', visible=False)
    for a in axs:
        tidy(a)
    fig.tight_layout()
    fig.savefig(FIG / 'fig5_attention.png', dpi=160)


def fig_listeners(M, preds, spk_preds, sup):
    y = preds['y']
    bins = [0, .2, .4, .6, .8, 1.01]
    mids = ['0–20%', '20–40%', '40–60%', '60–80%', '80–100%']
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={'width_ratios': [1.25, 1]})
    for p, pr in [('concat', [preds[f'concat_s{s}'] for s in SEEDS]), ('attn', [preds[f'attn_s{s}'] for s in SEEDS]),
                  ('concat_spknorm', spk_preds)]:
        acc = [np.mean([np.mean(q[(sup >= lo) & (sup < hi)] == y[(sup >= lo) & (sup < hi)]) for q in pr])
               for lo, hi in zip(bins[:-1], bins[1:])]
        axs[0].plot(range(5), np.array(acc) * 100, '-o', color=COL[p], lw=1.8, ms=5, label=LABEL[p].replace('\n', ' '))
    n = [int(np.sum((sup >= lo) & (sup < hi))) for lo, hi in zip(bins[:-1], bins[1:])]
    axs[0].set_xticks(range(5), [f'{m}\n(n={k})' for m, k in zip(mids, n)], fontsize=7.5)
    axs[0].set(title='Model accuracy vs. how many listeners heard the actor’s label',
               xlabel='share of audio-only listeners choosing the actor’s intended emotion', ylabel='accuracy (%)')
    axs[0].legend(frameon=False, fontsize=7)
    cats = [('clear_miss', 'listeners agree with actor;\nmodel does not'), ('matches_listeners', 'model agrees with listeners,\nnot actor'),
            ('neither', 'model agrees with neither')]
    colors = ['#e34948', '#2a78d6', '#b8b6ae']
    for i, p in enumerate(['concat', 'attn']):
        left = 0
        for (c, _), col in zip(cats, colors):
            w = 100 * np.mean([M[f'{p}_s{s}']['taxonomy']['error_categories'][c] for s in SEEDS])
            axs[1].barh(1 - i, w, left=left, color=col, height=0.55, edgecolor=SURF, linewidth=2)
            axs[1].text(left + w / 2, 1 - i, f'{w:.0f}%', ha='center', va='center', fontsize=8, color='white' if col != '#b8b6ae' else INK)
            left += w
    axs[1].set_yticks([1, 0], ['HW2 baseline', 'attention pool'])
    axs[1].set_xlim(0, 100)
    axs[1].set(title='What kind of error? (audio-only listener votes)', xlabel='% of the model’s errors')
    axs[1].legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors], [t for _, t in cats], frameon=False,
                  fontsize=6.8, ncol=3, loc='upper center', bbox_to_anchor=(0.45, -0.22))
    axs[1].grid(axis='y', visible=False)
    for a in axs:
        tidy(a)
    fig.text(0.01, 0.01, '*Speaker-normalised: features standardised with each speaker’s own clip statistics (transductive diagnostic, optimistic).',
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / 'fig6_listeners.png', dpi=160)


def fig_speakers(preds, spk_preds, sup, speakers):
    """Neutral clips that >= 60% of listeners heard as neutral: what does each test speaker get?"""
    y = preds['y']
    m = (y == 4) & (sup >= 0.6)
    spks = [s for s in np.unique(speakers[m])]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    colors = {'ANG': '#e34948', 'DIS': '#008300', 'FEA': '#6250d6', 'HAP': '#eda100', 'NEU': '#d6d4cd', 'SAD': '#2a78d6'}
    for ax, (title, pr) in zip(axs, [('HW2 baseline', preds['concat_s7']), ('baseline, speaker-normalised*', spk_preds[0])]):
        for i, s in enumerate(spks):
            mm = m & (speakers == s)
            cnt = np.bincount(pr[mm], minlength=6) / mm.sum() * 100
            left = 0
            for c in [4, 0, 1, 2, 3, 5]:
                ax.barh(i, cnt[c], left=left, color=colors[E[c]], height=0.72, edgecolor=SURF, linewidth=1.5)
                left += cnt[c]
        acc = np.mean(pr[m] == 4) * 100
        ax.set_title(f'{title}: {acc:.0f}% of these clips called neutral')
        ax.set_xlabel('% of the speaker’s clearly-neutral clips')
        ax.set_xlim(0, 100)
        ax.grid(axis='y', visible=False)
        tidy(ax)
    axs[0].set_yticks(range(len(spks)), [f'speaker {s}' for s in spks], fontsize=7.5)
    axs[0].invert_yaxis()
    axs[1].legend([plt.Rectangle((0, 0), 1, 1, color=colors[e]) for e in ['NEU', 'ANG', 'DIS', 'FEA', 'HAP', 'SAD']],
                  ['predicted neutral (correct)', 'anger', 'disgust', 'fear', 'happiness', 'sadness'],
                  frameon=False, fontsize=7, loc='lower right', bbox_to_anchor=(1.0, -0.32), ncol=6)
    fig.suptitle(f'Neutral clips that ≥ 60% of audio-only listeners heard as neutral (n = {m.sum()}, seed 7)',
                 fontsize=10, fontweight='bold')
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG / 'fig7_speakers.png', dpi=160)


def fig_architecture():
    fig, axs = plt.subplots(2, 1, figsize=(10, 4.6))
    def box(ax, x, y, w, h, text, fc, ec=INK2, fs=7.5, tc=INK):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc)
    def arrow(ax, x0, y0, x1, y1, col=INK2):
        ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='->', color=col, lw=1))
    for ax, title in zip(axs, ['HW2 baseline: position-specific read-out',
                              'HW3 design: attention pooling with a separate "where to look" scorer']):
        ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis('off'); ax.set_title(title, loc='left')
        for k in range(8):
            box(ax, 1 + k * 3.2, 19, 2.8, 7, f'x{k + 1}', '#eef0f3', fs=6.5)
        ax.text(14, 28.2, 'audio → trim → 8 equal segments → 88 features each', ha='center', fontsize=7, color=INK2)
        box(ax, 1, 9, 25.4, 6, 'shared MLP encoder f (88→128→64)\nsame weights for every segment', '#dbe7f8', fs=7)
        arrow(ax, 13.7, 19, 13.7, 15)
        for k in range(8):
            box(ax, 1 + k * 3.2, 1, 2.8, 5, f'z{k + 1}', '#c4d8f3', fs=6.5)
        arrow(ax, 13.7, 9, 13.7, 6)
    a = axs[0]
    box(a, 33, 1, 30, 5, 'concat [z1 … z8]  (512-d)', '#c4d8f3')
    arrow(a, 26.4, 3.5, 33, 3.5)
    box(a, 68, 1, 25, 5, 'linear 512→6: each position has\nits own weights for every class', '#f5e3c8', fs=6.8)
    arrow(a, 63, 3.5, 68, 3.5)
    a.text(68, 10, 'logit_y = b_y + Σ_k W_y,k · z_k', fontsize=7.5, color=INK2)
    a = axs[1]
    box(a, 33, 19, 30, 7, 'scorer g on relative view: loudness (log-RMS, MFCC-0)\ncentred on the utterance mean → s_k', '#f9e2b0', fs=6.6)
    arrow(a, 26.4, 22.5, 33, 22.5)
    box(a, 68, 19, 25, 7, 'α = softmax(s1 … s8)\nwhole-clip gain cannot move α', '#f9e2b0', fs=6.8)
    arrow(a, 63, 22.5, 68, 22.5)
    box(a, 33, 1, 30, 5, 'h = Σ_k α_k z_k  (64-d)', '#c4d8f3')
    arrow(a, 26.4, 3.5, 33, 3.5)
    arrow(a, 80, 19, 48, 6, col='#c98500')
    box(a, 68, 1, 25, 5, 'linear 64→6 (shared over positions)', '#f5e3c8', fs=6.8)
    arrow(a, 63, 3.5, 68, 3.5)
    a.text(33, 12.5, "variants: mean pool (α = 1/8), attention from z (s_k = v·tanh(A z_k)),\n"
           "attention + learned position embedding (z_k + p_k)", fontsize=6.6, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / 'fig1_architecture.png', dpi=160)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    M = json.loads((RES / 'analysis.json').read_text())['models']
    preds = np.load(RES / 'test_predictions.npz')
    spk = [np.load(RUNS / f'concat_spknorm_s{s}' / 'test_pred.npy') for s in SEEDS]
    votes = preds['votes']
    y = preds['y']
    sup = votes[np.arange(len(y)), y] / votes.sum(1)
    _, _, meta = load_data()
    speakers = meta['speaker'][preds['test_idx']]
    fig_architecture()
    fig_results(M, spk)
    fig_curves()
    fig_path_swap(M)
    fig_attention(M)
    fig_listeners(M, preds, spk, sup)
    fig_speakers(preds, spk, sup, speakers)
    print('figures in', FIG)


if __name__ == '__main__':
    main()
