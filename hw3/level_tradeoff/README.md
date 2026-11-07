# Experiment C: how much overall level should the model use?

[中文说明](README_zh.md)

Exploratory follow-up to the HW3 path diagnosis. The HW2 per-position concatenation read-out and the centred ("relative") input of model B are kept fixed. A separate branch adds back the two clip-level amplitude features (the utterance means of MFCC-0 and log-RMS), and during training that whole branch is switched off with probability p = 0, 0.25, 0.5 or 0.75. Each condition uses seeds 7, 11 and 23.

Read the [pre-run protocol](protocol-zh.txt) first (Chinese; it is kept unchanged because its SHA-256 is recorded in `results/config.json` and checked by `scripts/verify_results.py`). Results are stored separately from the protocol, so interpretations made after seeing results are not presented as predictions.

## Files

- `experiment.py`: two-branch model, training, validation-based selection, testing, and the intervention that moves only the level input while the relative branch is held fixed.
- `baseline_model.py`, `features.py`: copies of the HW2 model and feature code, kept so that the recorded source hashes still match.
- `summarize.py`: aggregates the three seeds and draws the validation and test curves.
- `results/config.json`: data, code and protocol hashes and the actor split. `results/selection.json`: the validation choice of p, written before test scoring.
- `results/val-summary.csv`, `results/test-summary.csv`: overall accuracy, macro-F1, recall for each of the six classes, and gain flip rates. Class values are recall, not precision.
- `results/*-predictions.npz`: per-clip class probabilities with their indices.

## Run

Use the same Python environment as HW2 (`requirements.txt`). The repository already contains the feature cache and the six A/B checkpoints; the WAV files can be downloaded with `scripts/download_crema.py`. Run from the repository root:

```bash
python3 hw3/level_tradeoff/experiment.py \
  --features data/features_K8.npz \
  --audio data/crema \
  --pilot hw3/runs \
  --out hw3/level_tradeoff/reproduction
python3 hw3/level_tradeoff/summarize.py \
  --out hw3/level_tradeoff/reproduction \
  --features data/features_K8.npz
```

The experiment refuses to overwrite an existing output folder; use a new `--out` for a rerun and pass the same `--out` to the summary script. The summary script reads the feature path from `config.json` by default; pass `--features` after moving to another machine.

## How to read the comparison

A and B are the already-trained original concatenation model and the input-centring model. C keeps B's input processing and per-position read-out and adds a bias-free 2→16→6 level network with a 64→1 gate. The gate reads only the relative representation. During training the entire level contribution of a clip is dropped, without inverted-dropout rescaling; at inference it is switched on.

p is a training intervention, not a measure of how much the model actually relies on the branch. For every C model we also move only the level input while holding the relative branch fixed, and compare inference strengths λ = 0, 0.25, 0.5, 0.75 and 1 on the validation set. C with λ = 0 is not equivalent to the independently trained B.

The ideal shift for ±6 dB is about ±0.6908 in natural-log RMS and about ±67.8823 in the current 128-mel MFCC-0, each divided by its training standard deviation before entering the model. This idealised level intervention is stored separately from the real waveform-gain re-extraction.

## Limits

Earlier test results had already motivated the class focus and the design. Here p was chosen on validation only, but the test results remain exploratory evidence on the same dataset. Seed standard deviations are not confidence intervals over new speakers. Non-monotonic results across the fixed grid and all classes are reported.

The waveform gain check reuses the clean F0 track, trimming and segment boundaries, without clipping, as in the earlier pilot; it does not represent all devices, noise or speaker differences. Centring is not full speaker normalisation. The two amplitude features are not equivalent, and using level information does not necessarily cause top-1 flips.

AI assistance was used for implementation, running and analysis. Neither the two-branch design nor channel dropout is claimed as a new method; the contribution is a hypothesis about this baseline, a controlled comparison, and traceable results.
