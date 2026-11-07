# HW3 — Speech Emotion Recognition

Can a small speech emotion model become less sensitive to recording volume without losing useful emotional cues?

[HW2 repository](https://github.com/Georg11a/deep-learning-assignment2) · [HW2 site](https://georg11a.github.io/deep-learning-assignment2/) · [Report](docs/HW3_REPORT_EN.md) · [Data attribution](DATA_ATTRIBUTION.md)

## Task and baseline

Predict six actor-intended emotions from **7,442 CREMA-D recordings**. The HW2 baseline divides each recording into eight segments, extracts 88 acoustic features per segment, and applies a shared **88 → 128 → 64 MLP**. The eight representations are concatenated for six-class prediction; no transcript is used.

Training, validation and test speakers are disjoint (64 / 9 / 18). Main results average seeds 7, 11 and 23 on 1,475 test recordings. Training uses SGD for 60 epochs; checkpoints are selected by validation accuracy.

![Baseline and attention architectures](hw3/figures/fig1_architecture.png)

## Experiments and results

- **Initial hypothesis:** attention pooling and a scorer with centered amplitude features might improve classification and gain robustness. Accuracy fell to about 51.5%, and gain sensitivity remained.
- **Diagnosis and revision:** independently swapping representations and attention weights implicated the representation path. **B** centers MFCC-0 mean and log-RMS mean within each utterance, before the encoder.
- **Follow-up C:** add an overall-level branch back to B and drop its contribution during training with probability *p*, testing how channel availability affects gain sensitivity.

| Model | Test accuracy | Mean ±6 dB prediction flip rate |
|---|---:|---:|
| A: HW2 baseline | 54.73% | 30.69% |
| B: input centering | 54.51% | **0.034%** |
| C: p = 0 | 54.69% | 10.40% |
| C: p = 0.25 | 55.46% | 1.12% |
| C: p = 0.50 | 54.35% | 0.34% |
| C: p = 0.75, validation-selected | 55.03% | 0.09% |

A flip means the same model changes its prediction after a volume adjustment; it is **not** an accuracy drop.

**Main lesson:** input centering nearly eliminates these gain-induced flips with little observed accuracy cost. Increasing *p* reduces C's sensitivity, but C has no established accuracy advantage over B (difference +0.52 percentage points; actor-bootstrap 95% interval −0.52 to +1.47).

B and C were motivated by earlier test inspection and remain exploratory. These experiments do not establish robustness to all recording conditions or removal of speaker bias. The contribution is controlled diagnosis and failure analysis, rather than a new normalization method.

## Extensions

- **[Personal references](hw3/personal_reference/README.md):** on a separate 1,421-query evaluation, using three neutral enrollment clips per speaker improved actor-average accuracy from 54.42% (retrained B) to 56.41% (amplitude + pitch reference). Enrollment clips are excluded from training loss and scoring; gain robustness was not tested.
- **[Listener labels](research/listener_labels/README-zh.md):** training on full listener votes reduced listener cross-entropy from 2.15 to 1.30, while actor accuracy fell from 56.9% to 27.9%.
- **[Audio-LLM pilot](research/omni_pilot/README.md):** with identical audio, changing only written loudness flipped 21.8% of answers. Acoustic descriptions improved actor accuracy but not listener-distribution matching.

The latter two studies are separate exploratory follow-ups. All reuse previously examined test speakers. **Actor intent and listener perception are different evaluation targets.**

## Run

From the repository root (Python 3.9+):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_results.py
```

To retrain the attention variants without overwriting saved runs:

```bash
python hw3/train_variants.py --runs-dir hw3/reproduction_runs
```

The feature cache, HW3 checkpoints and predictions are included; raw audio is not. See **[full experiment details and reproduction commands](docs/EXPERIMENT_DETAILS.md)** for the C sweep, personal-reference runs, audio download and tested environment. Implementation is in `training/` and `hw3/`; later studies are in `research/`.

AI assisted implementation, execution and drafting; the author is responsible for the design and interpretation.
