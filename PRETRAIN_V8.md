# PetLM v8 97M Chinese pretraining

## Result

The v8 base model is trained from random weights. It does not import an existing model's weights.

| Item | v7 | v8 frozen | Change |
|---|---:|---:|---:|
| Parameters | 33,526,784 | 97,700,352 | 2.91x |
| Layers / width / context | 8 / 512 / 256 | 12 / 768 / 512 | larger model and context |
| Train tokens | 26,485,110 | 50,683,232 | +91.4% |
| Accepted Wikipedia documents | 5,734 | 17,806 | 3.11x |
| Human-authored dialogue train rows | 766 | 851 | +11.1% after global holdout-first deduplication |

The 851 dialogue rows are the global prompt-deduplicated union of pinned OpenAssistant 1 and 2
Chinese training data. The source revisions and licenses are recorded in the snapshot manifests.
Source human feedback is not represented as a new local item-by-item human audit.

## Immutable snapshots

- Prepared corpus: `data/petlm/prepared-chinese-v8-50m-frozen`
- Token corpus: `data/petlm/tokens-chinese-v8-50m-frozen`
- Human dialogue: `data/petlm/dialogue/prepared-human-openassistant-v1`
- Configuration: `configs/pet-chinese-v8-97m.json`

Final token counts:

- train: 50,683,232
- validation: 3,006,891
- test: 2,838,733
- tokenizer SHA-256: `6492f3244360b202a98a7518501fe36fcd561037ddd8acd13211d5d081b5adc7`
- train binary SHA-256: `49f1c81f45c25019f9025eb18417166f1fa9707fa45c2b81808e968b6c6e291b`

## Wikipedia expansion

The added records are sequential eligible Chinese Wikipedia main-namespace pages from the pinned
Wikimedia `zhwiki` 2026-11-01 dump. The local samples cover eligible offsets 8,000 through 24,999.
Every sample has a sidecar metadata file containing its exact dump URL, offset, retrieval time and
SHA-256. Database import performs exact normalized-text deduplication before corpus cleaning.

## Holdout policy

OpenAssistant 1 and 2 are merged by normalized prompt before pretraining. A duplicate test prompt
wins over validation and training; validation wins over training. The final prepared corpus uses
only the resulting 851-row merged training file. The merged 92-row validation and 122-row test sets
remain outside training.

## GPU validation

The 97,700,352-parameter configuration completed forward, backward, optimizer and checkpoint-save
steps on the RTX 5060 Laptop GPU with BF16. Peak allocated CUDA memory was 2,763 MiB using batch 4,
gradient accumulation 4 and context 512.

## Training

Start a new random-weight run:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.run train `
  --config configs/pet-chinese-v8-97m.json `
  --tokens data/petlm/tokens-chinese-v8-50m-frozen `
  --output data/petlm/models/chinese-v8-97m `
  --max-seconds 600
```

Resume an interrupted run with the exact same schedule and dataset:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.run train `
  --config configs/pet-chinese-v8-97m.json `
  --tokens data/petlm/tokens-chinese-v8-50m-frozen `
  --output data/petlm/models/chinese-v8-97m `
  --resume data/petlm/models/chinese-v8-97m/latest.pt `
  --max-seconds 600
```

The 12,000-step base pretraining schedule processes about 98.3 million sampled tokens. Base
pretraining alone does not make the model a reliable assistant; instruction dialogue fine-tuning
and frozen blind evaluation must follow after this schedule is complete.

## Current checkpoint

The first bounded training phase completed 1,829 of 12,000 steps and saved:

- checkpoint: `data/petlm/models/chinese-v8-97m/latest.pt`
- report: `data/petlm/models/chinese-v8-97m/report.json`
- sampled tokens processed: 14,983,168
- validation loss: 6.8582 at step 500, 6.3746 at step 1,000, 6.0662 at step 1,500
- peak allocated CUDA memory: 2,766 MiB

This is an incomplete base-pretraining checkpoint. Continue it with the resume command above; do
not evaluate conversational quality or start dialogue fine-tuning until the base schedule finishes.
