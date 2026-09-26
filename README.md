# Amazon ML Challenge 2026: Business Entity Resolution

For every Source 1 business record, the pipeline finds its matching records in Source 2 and Source 3. It is scored by macro F0.5 per Source 1 entity.

| Version | Held-out validation | Public leaderboard |
|---|---|---|
| Prototype 1: single LightGBM | 0.9743 | 0.967 |
| **Prototype 2: two-stage + sibling features** (current code) | **0.9842** | **0.969** |

The top of the leaderboard was 0.9906 at handoff (2026-09-26).

## Start here
- **[business_entity_resolution/PROGRESS.md](business_entity_resolution/PROGRESS.md)** is the handoff note: what was built, what each change was worth, the leaderboard probes, and a ranked list of next ideas. Read its "Leaderboard" section first. Validation rose by 0.010 from prototype 1 to 2, but the leaderboard only rose by 0.002. That train/test shift is the main open problem, and there is a ready-made experiment to diagnose it.
- **[business_entity_resolution/README.md](business_entity_resolution/README.md)** covers how to run the pipeline, the step-by-step outputs and the source files.
- **[Documentation_template.md](Documentation_template.md)** is the methodology write-up for the final submission package. The team fields are still placeholders.

## Repository layout
```
README.md                        this file
Documentation_template.md        methodology write-up (goes into the submission zip)
business_entity_resolution/
  src/                           all pipeline code (entry point: run_all.py)
  README.md                      how to run, step by step
  PROGRESS.md                    handoff notes, results, ideas
  requirements.txt               pinned Python 3.13 dependencies
```

Not in the repo (see `.gitignore`):
- `student_resource/`: the challenge data and the official validator.
- `business_entity_resolution/work/`: intermediate files, about 16 GB after a full run.
- `business_entity_resolution/output/`: submission TSVs and leaderboard probe files.
- The submission zips.

## Setup
1. Put the challenge files next to the code:
   ```
   <root>/student_resource/dataset/{train,test}/...
   <root>/student_resource/utils/validate_submission.py
   ```
   Or point `BER_DATA` at the dataset directory.
2. Install the dependencies:
   ```
   pip install -r business_entity_resolution/requirements.txt
   ```
3. Run the pipeline:
   ```
   cd business_entity_resolution/src
   export PYTHONIOENCODING=utf-8     # Windows cmd: set PYTHONIOENCODING=utf-8
   export BER_JOBS=5                 # worker processes/threads
   python run_all.py
   ```
   A full run takes about 7 hours on a 12-core, 16 GB, CPU-only machine; blocking is about 3 hours of that. Peak memory is about 10 GB. Every stage writes its output to `work/`, so after a crash you rerun only the unfinished step. After a full run, stage 2 alone can be retrained in about 35 minutes with `python train.py --stage2`.

## Approach in one paragraph
Each S2/S3 record chooses at most one S1 parent. Candidates come from a country-partitioned TF-IDF search with two channels (combined name+address, and address-only), using transliteration, abbreviation and French region normalisation. Stage 1 is two LightGBM models on string-similarity and candidate-competition features, each scoring the half of the training records it did not see. **Sibling expansion** adds candidates through records that are already confidently matched. The key data insight is that copies of one business share typos, spacing and unit numbers. **Stage 2** adds probability-competition and sibling-similarity features. Finally, each S1 entity keeps the number of matches that maximises its expected F0.5. No external data or pretrained models are used.
