# Where should amplitude information enter a speech emotion model?

*Editable report source. Page count must be checked after export. The personal-reference study is documented separately.*

## Baseline and original hypothesis

The HW2 system predicts six actor-intended CREMA-D emotions from eight segments of 88 acoustic features. A shared 88→128→64 MLP encodes each segment; ordered concatenation feeds a six-class linear head. No transcript is used. The speaker-disjoint split contains 64/9/18 training/validation/test actors. Three seeds use the same SGD recipe, 60-epoch budget and validation checkpoint selection.

The initial hypothesis was that adaptive attention pooling could select useful segments more flexibly than position-specific concatenation. A separate scorer receiving all 88 features, with MFCC-0 mean and log-RMS mean centered within the utterance, was intended to reduce sensitivity to global gain. This was a hypothesis about aggregation and robustness, not a claim that loudness determines emotion.

## Observations and exploratory revisions

Attention reduced mean test accuracy from 54.7% to 51.5%; relative scoring did not solve gain sensitivity. Position embeddings recovered performance to 53.1%, consistent with useful position information but not a unique explanation of the loss.

A 2×2 diagnostic independently exchanged clean/perturbed representations and attention weights. Among ordinary-attention correct→wrong events under global gain, representation-only replacement was sufficient in 83.8% and weight-only replacement in 15.8%. These potentially overlapping sufficiency rates are not causal percentages. They motivated an input intervention: **B** centers the same two columns before the encoder, retaining the HW2 readout.

**C** then separates those two utterance means into a small 2→16→6 branch, gated by the relative representation. During training, its entire contribution is dropped with probability p∈{0,.25,.5,.75}; inference normally enables it. This follow-up protocol was recorded before its runs. All p values have the same architecture; three-seed mean validation accuracy selected p=.75 before new test scoring. Earlier test observations had motivated the research, so these remain exploratory follow-ups.

## Results and interpretation

Accuracy is on original recordings. Flip rate averages top-1 changes under −6 and +6 dB for each model.

| Model | Test accuracy | Mean gain flip |
|---|---:|---:|
| A: original | 54.73% | 30.69% |
| B: input centering | 54.51% | 0.034% |
| C: p=0 | 54.69% | 10.40% |
| C: p=.25 | 55.46% | 1.12% |
| C: p=.5 | 54.35% | 0.34% |
| C: p=.75, validation selected | 55.03% | 0.09% |

Within the tested grid, p controls gain sensitivity much more clearly than accuracy. Selected C exceeds B by .52 percentage points; the descriptive paired actor-bootstrap 95% interval [−.52,1.47] does not establish a reliable improvement. B is the simpler practical correction. Its three flip events in 8,850 comparisons rule out an exact zero-flip claim.

Two interventions qualify the mechanism. First, moving only the level input along the theoretical gain direction closely reproduces C's waveform-gain sensitivity. The shifts are feature-specific: ±67.88 for MFCC-0 and ±.6908 for log-RMS before division by training standard deviations. Second, disabling the level contribution (λ=0) in the frozen p=.75 model changes validation accuracy from 59.43% to 59.47%. Consequently, differences from independently trained B cannot be attributed solely to using absolute level at inference. For p=0, disabling the branch does reduce overall validation accuracy. Happy-class differences are exploratory; selected C's happy recall is 62.4% on validation versus 41.5% on test.

## Contribution, limitations and next step

The contribution is a controlled case study of intervention location and channel use, not a new normalization method. Attention's failure motivated a simpler effective correction; C showed that training availability and inference reliance must be checked separately. Gain tests hold clean trimming, boundaries and F0 fixed and avoid clipping; they do not establish robustness to all devices, speakers or natural emotions. Actor intent and listener perception are distinct targets.

A separately frozen extension used three neutral enrollment recordings per actor. D3 achieved 56.41% actor-average accuracy versus retrained B's 54.42% on their common query set. A post-hoc check retains a +2.05-point D3−B difference [0.35,3.68] on 1,151 queries not sharing enrollment sentence types. However, D3's neutral→sad rate remains 18.43%, equal to A; listeners also show the negative loudness–sadness association. Fixed personal references do not cancel query-only gain shifts. This enrollment-assisted, same-corpus extension neither isolates identity from session effects nor establishes causal bias removal. Details are in `hw3/personal_reference/`; these results must not be mixed with the main table.

AI assisted implementation, execution and drafting. Final design interpretation remains the student's responsibility. Data: [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D). Code and results: this repository.
