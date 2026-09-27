# Business Entity Resolution — ML Challenge 2026

This pipeline finds, for every Source 1 business record, its matching records in Source 2 and Source 3. The steps are: data → normalisation → blocking → pair features → stage-1 ensemble (LightGBM + XGBoost + CatBoost) → sibling expansion → stage-2 ensemble → per-entity match selection. Predictions from all three model types are blended with per-stage weights optimised on validation.

## Setup
Requires Python 3.13. The full pipeline was run on Windows 11 (12 cores, 16 GB RAM, CPU only). With `BER_JOBS=5` an end-to-end run takes about 7 hours: normalisation ~15 min, blocking ~3 h, pair features ~55 min, training ~1 h, prediction ~1 h. Peak memory is about 10 GB.
```
pip install -r requirements.txt
```
By default the pipeline expects this layout:
```
<root>/student_resource/dataset/{train,test}/...   # challenge data
<root>/business_entity_resolution/src/...           # this code
```
To use other locations, set `BER_DATA` (the dataset dir), `BER_WORK` (intermediate files) and `BER_OUT` (output dir). `BER_JOBS` sets the number of worker processes and threads (default: at most 5). `BER_DEVICE` selects model training on `cpu` (default) or `cuda`.

To train the ensemble on an NVIDIA GPU, set `BER_DEVICE=cuda` before running `train.py` or `run_all.py`. In Windows PowerShell, use `$env:BER_DEVICE="cuda"`; in Command Prompt, use `set BER_DEVICE=cuda`. CUDA must be available to Python. XGBoost 2.0 or newer needs CUDA support, CatBoost needs GPU support, and LightGBM needs a build compiled with its CUDA backend (the standard wheel may not include it). The data preparation, blocking, and feature-generation stages remain CPU-based, so this accelerates model fitting, not the entire pipeline. CatBoost omits two CPU-specific training options in CUDA mode.

## Reproduce end-to-end
```
cd src
set PYTHONIOENCODING=utf-8          # (bash: export PYTHONIOENCODING=utf-8)
python run_all.py
```
`run_all.py` runs these steps in order; each can also be run on its own:

| Step | Script | Output (in `work/`) |
|---|---|---|
| 1. Learn Indic→Latin token dictionary from train labels | `learn_translit.py` | `translit.json` |
| 2. Normalise names and addresses | `prep.py train`, `prep.py test` | `<split>_source{1,2,3}.parquet` |
| 3. Blocking: sparse TF-IDF top-K per S2/S3 record (combined + address-only channel), same country | `block.py train`, `block.py test` | `<split>_cand_raw.parquet` |
| 4. Pruned candidate set + pair features (+ labels for train) | `build_pairs.py train`, `build_pairs.py test` | `<split>_cand_final.parquet`, `<split>_pairs.parquet` |
| 5. Stage 1 (two half-split models), sibling expansion, stage 2; threshold for macro F0.5 | `train.py` | `model_s1_{0,1}.txt`, `model_s2.txt`, `threshold.json` |
| 6. Score test (same stages) and write both TSVs | `predict.py` | `output/matching_results.tsv`, `output/candidate_pairs.tsv`, `test_prob.parquet` |
| 7. Per-entity expected-F0.5 match selection | `postprocess.py` | rewrites `output/matching_results.tsv` |

`train.py --stage2` retrains only stage 2, reusing the stage-1 models, their out-of-fold probabilities and the expansion pairs.

`analyze_blocking.py` is optional. It reports blocking recall on the training split.

Held-out validation (S1 fold 0, every candidate of every record touching it): **macro F0.5 = 0.9839** with the stage-2 threshold, **0.9842** after `postprocess.py` (stage 1 alone: 0.9783; prototype 1: 0.9743).

Then validate the output:
```
cd ../../student_resource
python utils/validate_submission.py --matching ../business_entity_resolution/output/matching_results.tsv \
    --candidate ../business_entity_resolution/output/candidate_pairs.tsv --test-dir dataset/test
```

## Source files
- `config.py`: paths and settings.
- `normalize.py`: transliteration, accent folding, abbreviation and state canonicalisation, number clean-up, consonant-skeleton key.
- `learn_translit.py`: learns the Indic→Latin token dictionary from co-occurrence in training pairs.
- `prep.py`: normalises records in parallel and writes parquet.
- `block.py`: feature-hashed TF-IDF over name and address tokens, with top-K search by `sparse_dot_topn`.
- `features.py`: rapidfuzz string similarities plus candidate-competition (context) features.
- `build_pairs.py`: prunes to the final candidate set, adds labels, fold split and features.
- `expand.py`: sibling expansion. A record without a confident match gets as extra candidates the Source 1 entities of confidently matched records that share its normalised address, core name, name skeleton or house number + street.
- `stack.py`: stage-2 features: competition over stage-1 probabilities, sibling features (how a record compares with the other records confidently assigned to the same entity: copies of one business share typos, spacing and unit numbers).
- `ensemble.py`: XGBoost and CatBoost fit/predict/save/load wrappers, hyperparameters, and blending utilities.
- `train.py`: two-stage ensemble (LightGBM + XGBoost + CatBoost) training with blend weight optimisation and macro-F0.5 threshold search.
- `predict.py`: scores test pairs through the same stages and writes the two TSVs.
- `postprocess.py`: per-entity choice of how many matches to keep, by expected F0.5.

No external data, APIs or pretrained models are used. The learned models are LightGBM (MIT), XGBoost (Apache 2.0) and CatBoost (Apache 2.0).
