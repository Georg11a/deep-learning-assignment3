# Audio-LLM reference pilot (Qwen2.5-Omni-7B)

Zero-shot. Each test clip is given to the model with, depending on the condition, a short block of automatically measured acoustic cues. The model answers a 6-way multiple choice; we read the next-token probabilities of the letters A–F (one forward pass per clip × condition). Everything runs from `03_Omni_Reference_Pilot.ipynb` on one A100 (UVA HPC, Open OnDemand JupyterLab), from the repository root.

| Condition | Text given with the audio |
|---|---|
| `audio_only` | none |
| `global` | loudness and pitch level vs other speakers, plus within-clip variation |
| `within` | within-clip variation only |
| `personal` | loudness and pitch level vs this speaker's own neutral voice (lower / about the same / higher), plus within-clip variation |
| `cf_loud_low`, `cf_loud_high` | `global` text with the loudness level forced to low / high; audio unchanged |

## Results (all 1,421 test queries)

| Condition | Actor accuracy | In listeners' top vote | Cross-entropy to votes ↓ | Neutral → sad |
|---|---:|---:|---:|---:|
| audio only | 37.3% | 55.2% | 1.796 | 0.0% |
| global | 46.1% | 50.9% | 1.849 | 6.8% |
| within | 38.2% | 44.2% | 2.037 | 1.2% |
| personal | 49.1% | 35.4% | 2.410 | 15.5% |

Counterfactual, identical audio: changing only the written loudness from low to high changes the answer for 21.8% of clips; P(anger) +0.10, P(sadness) −0.07, P(happiness) +0.04.

Reading:
- Listening alone, the model behaves more like a listener than like the actor (it often answers neutral).
- Written cues raise actor accuracy, mostly through the overall-level cues, while moving answers away from listeners. The own-voice wording shifts many answers toward happiness.
- The model clearly uses the written evidence, and reads written loudness as anger versus sadness.

Limits: single zero-shot run, no intervals yet; option order is fixed (compare conditions rather than absolute accuracy); about 20% of next-token probability falls outside the letters A–F (`letter_mass` 0.76–0.83; section 7b of the notebook inspects it); the `personal` and `global` conditions use different wording, so reference and vocabulary are confounded; a shuffled-evidence control is still needed to separate the content of the text from its presence. The personal-reference bucketing and wording were revised after an exploratory pilot (the first version described clips at the speaker's own neutral level as "low"); the final thresholds are estimated from training speakers only and were fixed before the reported run.

## Files

- `03_Omni_Reference_Pilot.ipynb`: the notebook used for the reported run.
- `results/`: copy here from the HPC run folder (`hw3/omni_pilot/<RUN_NAME>/`): `manifest.json`, `summary.json`, `predictions.csv`, `summary.png`. The manifest records the model, prompt version, cut points, package versions and GPU.
