# PetLM v8 completion and dialogue evaluation

Date: 2026-11-04

## Base pretraining

- Initialization: random weights; no existing model checkpoint imported.
- Model: 97,700,352 parameters, 12 layers, width 768, context 512.
- Dataset: `data/petlm/tokens-chinese-v8-50m-frozen`.
- Train tokens: 50,683,232; validation: 3,006,891; test: 2,838,733.
- Schedule: 12,000 / 12,000 steps complete.
- Sampled tokens processed: 83,320,832.
- Final base validation loss: 4.2460; at step 2,000 it was 5.8296.
- Peak allocated GPU memory: about 2.69 GiB.
- Base checkpoint: `data/petlm/models/chinese-v8-97m/latest.pt`.

## Dialogue SFT

The immutable SFT snapshot is `data/petlm/dialogue/prepared-pet-human-v11`:

- Train: 4,708 rows; validation: 210; frozen test: 122.
- Train groups: 3,913 programmatically composed pet scenarios, 30 legacy user-confirmed pet examples,
  795 globally deduplicated natural OpenAssistant rows (including 117 rule-assisted audit rows).
- Prompt overlap across train, validation and test: zero after normalization.
- Whole conversation trees were isolated with test > validation > train priority. 112 training and
  9 validation rows from trees held out by another source were excluded.
- The 3,913 synthetic examples are not human-authored. Rule-assisted audits are not a human
  item-by-item review.

Fine-tuning started from the completed v8 base checkpoint, used assistant-only reply loss, learning
rate 1e-5, batch size 2, accumulation 2, no legacy replay, and stopped after the 1,000-step schedule.
The best checkpoint was step 850:

- Parent validation reply loss: 4.2754.
- Best SFT validation reply loss: 3.5256.
- SFT checkpoint: `data/petlm/models/dialogue-chinese-v8-v11/best.pt`.
- Report: `data/petlm/models/dialogue-chinese-v8-v11/report.json`.

## Frozen evaluation

All generations used the model-only path, temperature 0.3, seed 42. Test prompts and reference
answers were not supplied to training.

| Evaluation | Base v8 | SFT v8-v11 |
|---|---:|---:|
| 122 frozen human-authored test replies, token loss | 3.9939 | 3.9396 |
| 22 independent natural-reference replies, token loss | 5.4218 | 4.8889 |
| 500 frozen pet blind cases, heuristic minimum-element pass | 0/500 | 13/500 (2.6%) |

The 500-case answer key is generated and the scoring is keyword/result based; this is a coarse
automated check, not blinded human evaluation. The 22 natural-reference answers have rule-assisted
item-level program review, not human item-level review. Despite lower reply loss and a small change
in the 500-case heuristic score, generated outputs remain semantically unreliable. Inspection of
the 22 natural prompts found off-topic responses, malformed reasoning and repetition. The model is
not ready to be presented as a fluent or reliable conversational pet.

Generated outputs and reports:

- `data/petlm/evaluations/chinese-v8-base-final-blind-v3.jsonl`
- `data/petlm/evaluations/chinese-v8-base-final-blind-v3.score.json`
- `data/petlm/evaluations/dialogue-chinese-v8-v11-final-blind-v3.jsonl`
- `data/petlm/evaluations/dialogue-chinese-v8-v11-final-blind-v3.score.json`
- `data/petlm/evaluations/dialogue-chinese-v8-v11-natural-blind-v3.jsonl`

Conclusion: the full training and evaluation workflow completed, but the blind results do not
support the claim that this model can communicate smoothly. Next improvements should prioritize a
larger, explicitly human-reviewed pet dialogue corpus and more capable dialogue modeling; do not
raise sampling or overfit these small snapshots to inflate scores.
