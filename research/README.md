# Exploratory follow-ups (not part of HW3)

Two studies run after the HW3 experiments. They reuse the HW3 data, speaker split (64 / 9 / 18 actors) and the neutral enrollment clips (DFA, TAI, IOM) of `hw3/personal_reference/`. The 18 test speakers had already been used throughout HW3, so all results here are exploratory.

| Folder | Question | Training? |
|---|---|---|
| `listener_labels/` | Same small model and input; what changes when the training target is the actor label, the listeners' top vote, or the full listener vote distribution? | Yes, CPU, about 3 minutes |
| `omni_pilot/` | Does an audio LLM (Qwen2.5-Omni-7B) change its judgement when the same acoustic cue is described against different references? Does it follow written evidence? | No (zero-shot), one A100 GPU |

Listener votes are CREMA-D's audio-only ratings (`data/tabulatedVotes.csv`, rows whose id starts with 1; median 9 raters per clip).
