# Personal reference: an exploratory follow-up

Read [the frozen protocol](protocol-zh.txt) before the results. Every actor supplies the same three sentence types (DFA, TAI, IOM), with actor-intended neutral labels. Enrollment recordings are excluded from **all** training losses and evaluation sets. This is enrollment-assisted inference, not zero-shot inference or unlabeled adaptation.

## Conditions

All models use the same SegmentMLP (22,726 parameters), three seeds, and 60-epoch recipe. A and B are **retrained** with the smaller query set; historical results cannot be directly compared.

| Condition | Transformation after common train-query z-score |
|---|---|
| A | None |
| B | Center MFCC-0 mean and log-RMS mean within the current utterance |
| D2 | Subtract the actor's enrollment mean from those two columns |
| D3 (fixed primary) | D2 plus personal centering of log-F0 mean |

For D2/D3, the frozen model is also evaluated with five cyclic, same-split, non-self reference assignments. Wrong-reference scores average the five evaluations; they are **not** predictions from an ensemble of five probability vectors.

## Results

1,421 test queries, 18 test actors, 3 seeds. Each actor's three enrollment clips are excluded. Accuracy below gives every actor equal weight (the predeclared primary metric).

| Condition | Actor-average accuracy | Sample accuracy | Macro-F1 | Wrong-reference actor accuracy |
|---|---:|---:|---:|---:|
| A | 54.75% | 54.75% | .5390 | — |
| B | 54.42% | 54.42% | .5359 | — |
| D2 | 56.49% | 56.49% | .5575 | 53.87% |
| D3 | 56.41% | 56.42% | .5550 | 52.64% |

D3 minus B: +1.99 percentage points, descriptive paired actor-bootstrap 95% interval [0.21, 3.66]. D3 matched minus wrong reference: +3.78 points [2.24, 5.34]. These support a benefit of the matched calibration setup on this split, not proof of isolated speaker identity or a guarantee on new corpora.

D3 minus D2: −0.07 points [−1.17, 1.13]. Adding a personal pitch reference has no established incremental test benefit, despite a larger validation difference. D3 remains the fixed primary condition; D2 is not selected retrospectively from the test ranking.

**Correction to the earlier discussion:** registration log-RMS and neutral-query sadness have correlations −.60 for A, −.37 for B, −.09 for D2 and −.17 for D3. Listener sadness vote fraction also correlates negatively (−.61; descriptive interval [−.82, −.24]). Thus the earlier claim that listeners lack this relationship is withdrawn. These are between-actor associations in a small acted corpus, not proof of a stereotype, a causal loudness rule, or people's true emotions. The smaller D3 correlation is not itself a formal test of change in correlation; marginal confidence-interval overlap is also not a paired test of that change. Neutral→sad error rates are 18.43%, 20.91%, 17.60%, 18.43% for A/B/D2/D3: **D3's rate did not decrease relative to A**. It is lower than B's point estimate, which is a different comparison.

## Post-hoc sentence-overlap check

Enrollment/query **recordings** are disjoint, but 270 queries use the same sentence types as enrollment (DFA/TAI/IOM, acted with other emotions). The remaining 1,151 queries use other sentence types. This is an explicitly post-hoc sensitivity check of already saved predictions, with no retraining and no change to the frozen protocol.

| Test subset | Queries | D2−B actor accuracy, points [95% interval] | D3−B actor accuracy, points [95% interval] |
|---|---:|---:|---:|
| All | 1,421 | +2.06 [0.12, 3.91] | +1.99 [0.21, 3.66] |
| Shares enrollment sentence | 270 | +0.37 [−2.23, 3.09] | +1.73 [−1.11, 4.69] |
| Does not share enrollment sentence | 1,151 | +2.46 [0.50, 4.34] | +2.05 [0.35, 3.68] |

These independently reproduced point estimates match the review. Intervals use 5,000 paired actor resamples with the repository's fixed seed 20261108; endpoints differ slightly from the review's supplied intervals, whose script/settings were unavailable locally. The observed benefit persists without an enrollment reading of the query sentence. This does **not** establish generalization to wording unseen during parameter training or remove all lexical/session confounds. See [JSON results](results/sentence-overlap-check.json) and [per-actor data](results/sentence-overlap-per-actor.csv).

## Gain expectation: fixed reference versus shared gain

For either amplitude column, write the standardized query value as x and its enrollment reference as r. D2/D3 use x−r. Under the ideal additive feature-shift approximation:

- Gain changes the query alone by δ: (x+δ)−r = (x−r)+δ. Fixed personal centering does not cancel this shift. The two input-column shifts match those entering A, but the trained models differ, so prediction changes need not match A.
- The same gain changes both query and all enrollment clips: (x+δ)−(r+δ) = x−r. The common shift cancels in those columns.

This is an analytical expectation, **not a measured D2/D3 gain result**. Feature extraction has floors, numerical effects and potentially other changes; different devices need not act as a single common gain. Future waveform tests should compare query-only gain with matched enrollment-and-query gain while holding all other choices fixed. No such new gain experiment was run in this review update.

## Reproduce

From the repository root, after installing requirements:

```bash
python hw3/personal_reference/run.py --check-only
python hw3/personal_reference/run.py --out hw3/personal_reference/reproduction
python hw3/personal_reference/summarize.py --out hw3/personal_reference/reproduction
python hw3/personal_reference/sentence_overlap_check.py --out hw3/personal_reference/reproduction
```

No WAV download is required when using the bundled feature cache. Existing result directories are protected. All four conditions are trained and validated before `frozen-before-test.json` is written and test queries are scored. Inspect `results/manifest.json` for enrollment/query filenames, `checks.json` for separation checks, and `config.json` for hashes and recipe.

## Boundaries

- Three same-corpus neutral performances approximate an enrollment reference; they do not establish a person's long-term normal voice. Neutral enrollment is an explicit extra resource.
- Test actors are absent from parameter training, but the same old test split inspired this research. This remains exploratory. The 18-actor bootstrap is descriptive, with fixed training data and three seeds.
- Actor identity, channel and session characteristics are not separated. Mismatched references can be out of distribution.
- This experiment does not test waveform gain robustness of D2/D3; do not assume they inherit B's near-invariance to query-only gain changes.
- No new listener annotation, soft-label training, RLHF, age/sex analysis or cross-corpus evaluation was performed.
- `1001_DFA_NEU_XX.wav` and `1002_DFA_NEU_XX.wav` are enrollment clips here. Do not use them to demonstrate D2/D3 evaluation predictions; they may only illustrate the earlier A/B experiments or enrollment itself, explicitly labelled.

Saved results include every query prediction, six-class recall, each actor's metrics, listener-agreement subsets, training histories, checkpoints, paired intervals and figures. No test-driven follow-up tuning was added.
