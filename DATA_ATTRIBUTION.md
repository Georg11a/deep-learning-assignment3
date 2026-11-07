# Data provenance and licensing

This research package uses **CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset**, Houwei Cao, David G. Cooper, Michael K. Keutmann, Ruben C. Gur, Ani Nenkova and Ragini Verma (2014). [Paper](https://doi.org/10.1109/TAFFC.2014.2336244) · [Official repository](https://github.com/CheyneyComputerScience/CREMA-D).

- `data/features_K8.npz`: a derived database of 8 × 88 acoustic summaries for 7,442 CREMA-D recordings, extracted by the HW2 feature code. It contains filenames and actor IDs, not waveform audio.
- `data/tabulatedVotes.csv`: the official `processedResults/tabulatedVotes.csv` table. Analyses use queryType 1 (audio-only), identified by integer row ID divided by 100000. The original table is retained so the selection can be inspected.
- `scripts/crema_files.txt`: the 7,442 source WAV filenames. `scripts/download_crema.py` downloads recordings directly from the official source. No raw audio is bundled.

The source database is licensed under the [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/), and individual contents under the [Database Contents License 1.0](https://opendatacommons.org/licenses/dbcl/1-0/). The derived feature database is provided under ODbL 1.0. These notices do not relicense original recordings or votes. Preserve attribution and applicable license obligations when redistributing these materials.

Actor labels describe intended acted emotions. Listener votes describe perceived emotions under the annotation setup. Neither directly measures a person's internal emotional state. This project does not infer age, gender, personality or real mental state from the recordings.

HW2 code provenance: [georg11a/deep-learning-assignment2](https://github.com/georg11a/deep-learning-assignment2). Core feature and baseline model files match the local HW2 source used for the experiments. No third-party code license is invented by this package; source licensing remains with its original authors.
