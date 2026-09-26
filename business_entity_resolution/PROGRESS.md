# Progress notes

## Prototype 2: complete (2026-09-26)
- **Validation: macro F0.5 0.9842.** Stage 2 at threshold 0.66 scores 0.9839, and `postprocess.py` adds 0.0003. Stage 1 alone scores 0.9783; prototype 1 scored 0.9743.
- **Leaderboard: 0.969.**
- **Test output:** 5,944,437 matches, 5.5% of S1 left empty. Candidates: 38.6M after pruning plus 1.30M from sibling expansion.
- **Validator:** PASS, including `--check-ids`.
- **Packages:**
  - `../submission.zip` holds prototype 2.
  - `../submission_v1.zip` is the old package.
  - `work/v2a/` holds the prototype-2 model, threshold, validation predictions, test probabilities and output TSVs.
- **Before the final package:** team fields in `../Documentation_template.md` are still placeholders. Rename the zip to `<team_name>_submission.zip`.

### What prototype 2 changed (validation effect)
| Change | Validation |
|---|---|
| Prototype 1 | 0.9743 |
| Multi-channel blocking: address-only top-6 channel, address skeletons, name char 4-grams, name numbers. Raw recall 97.74% → 98.58%. | |
| French region/department → region code; `N°`, EI/EARL/…, Ets/Cie; `PAS`, `bât` | |
| Number-set features for near-miss distractors (`num_r_extra`, `num_l_extra`, `num_jacc`) | |
| Stage 1 as two half-split models (out-of-fold probabilities) | |
| Sibling expansion (`expand.py`): 0.90M train and 1.30M test extra pairs | → 0.9783 (stage 1) |
| Stage 2 (`stack.py`): probability competition + sibling features | → 0.9839 |
| Per-S1 expected-F0.5 selection (`postprocess.py`) | → **0.9842** |

Tried and dropped:
- A third stage that recomputes siblings from stage-2 probabilities: −0.001.
- Sibling features relative to competing candidates: +0.0001, not kept.

## Leaderboard
- Prototype 1: **0.967** (validation 0.9743).
- Prototype 2: **0.969** (validation 0.9842). Validation rose by 0.010 but the leaderboard only by 0.002. **This gap is the main open problem.**
- Top of the leaderboard on 2026-09-26: 0.9906, 0.9894, 0.9888.

### Leaderboard probes (2026-09-26, uploads are unlimited)
The probe files are in `output/probes/` and can be regenerated from `work/test_prob.parquet`. Each probe is prototype 2 with one change:

| Probe | Leaderboard |
|---|---|
| France predictions blanked | 0.836 |
| India predictions blanked | 0.580 |
| US predictions blanked | 0.578 |
| France predictions taken from prototype 1 | 0.969 (no change) |

What the probes show:
- **France is not the cause.** Swapping in prototype 1's France predictions changes nothing.
- **The drops don't fit the test set's country mix.** Blanking a country should lower the score by roughly `weight × (country score − singleton rate)`. The US drop of 0.391 at its 38.3% share of test S1 would need a US score above 1. So either the leaderboard is computed on a subset with more US in it, or the scoring differs from our assumption.
- **The singleton rate is consistent.** The three drops together imply about 5.6% of scored S1 entities have no match, the same as in train.

**Unexplained shift between train and test.** Test has **5.75 S2/S3 records per S1 against 4.67 in train** (+23%, the same in every country). We predict about 3.4 matches per S1 on both. Two hypotheses:
1. Test S1 is missing about 19% of the entities whose copies remain in S2/S3. Those orphan copies look like confident matches to a same-name "twin" S1, causing false merges. Fix: train on data with S1 entities deliberately removed. A cheap validation simulation without re-scoring showed no effect, but that simulation can't capture the context features changing.
2. Test simply has more true copies per S1, so we under-match. Fix: recall-oriented changes.

The threshold probes `output/probes/C_thr_{35,50,80,92,97}.tsv` were built to decide between these but have **not been uploaded yet**:
- if the strict files score higher, hypothesis 1 holds;
- if the loose files score higher, hypothesis 2 holds;
- if they are flat, the cause lies elsewhere.

## Where the remaining score is (prototype-2 validation)
Total loss is 3,550 over 220k S1 entities, i.e. 0.016:

| Error type | Loss |
|---|---|
| Partial recall | 1,757 |
| Missed every match of an S1 | 1,028 |
| Some wrong merges | 600 |
| Wrong merge on an S1 with no matches | 159 |

- **Ceiling with a perfect model on our candidates: 0.9947.** Candidates cost about 0.005; the model costs about 0.011.
- **Missed true pairs:** 12.0k are not in the candidate set, 8.4k fall below the threshold, and 5.4k go to another S1.
- **61% of all missed true pairs have an empty address.** Among pairs lost to another S1, the share is 84%. Typically the name fits two S1 entities in different cities. The wrong winner is a real business with its own matches; only 3% are singletons.
- **About 8–12% of misses are invented names at the right address**, e.g. `veogild` or `korhalovantage` at the S1 address.
- **Wrong merges are mostly parentless near-miss distractors**: the same building with a different unit or house number, or a different name at the same address.
- **Weak twins are rare:** two missed siblings of one S1 with an identical raw address cover only 374 pairs.

## Brainstorm: what it would take to reach about 0.99+
Reaching 0.992 means removing about half of the remaining loss, which needs several of the ideas below. The leaders prove the data allows it, so they are exploiting structure we only partly use. Ranked by expected value:

1. **Collective, cluster-level resolution. Largest expected gain, about +0.002 to +0.004.**
   - **Why:** we still score pairs one at a time and link each record to its siblings only through confident ones, one hop at a time. The data is generated per entity, so every copy of a business is a noisy variant of one entity-level record.
   - **How:**
     - Build a record–record graph over S2/S3, with edges from the expansion keys (exact address, core name, skeleton, number+street) plus the high-similarity pairs we already score.
     - Form clusters, then compute cluster-level evidence for each candidate S1: sum and max of member probabilities, the fraction of members agreeing, and the best address in the cluster against S1. Then assign whole clusters, with one S1 per cluster.
     - Iterate twice so that a strong member lifts its weak, empty-address siblings.
   - **Cost:** no new blocking. It reuses the existing candidates, probabilities and the stage-2 retrain (`train.py --stage2`, about 35 min plus 1 h prediction).
   - **Risk:** spurious clusters for generic names. Cap clusters by key frequency, as `expand.py` does.
2. **Candidate recall for the last 12k pairs. About +0.001 to +0.002; the ceiling is 0.9947.**
   - Run a second expansion round using stage-2 probabilities: more confident records means more usable siblings.
   - Add a raw-name key that keeps the original spacing and punctuation, since siblings share these quirks.
   - Use fuzzy keys, e.g. an address skeleton with house number, and a name 4-gram MinHash.
   - For empty-address records, raise the name-only top-K. Many same-name S1 entities can crowd the true one out of the top 12.
3. **Near-miss numbers. About +0.0005 to +0.001.** Real noise and distractors change numbers differently:
   - Real noise truncates, pads or adds a suffix letter (`7402`→`740`, `882`→`00882`, `10216`→`10216b`).
   - Distractors substitute a different number (`315`→`320`).

   Add digit-level edit-type features: prefix or suffix truncation, zero padding, suffix letter, absolute numeric difference, and the number of differing digits. Apply them to the house number and to each unit number.
4. **Invented-name detector. About +0.0005.** Train a character n-gram model on S1 names. An S2/S3 name that is unlike any real vocabulary, combined with a strong address match, suggests a DBA-style true match. A real-looking different name at the same address (`Mumbai Adworks` vs `Mumbai Exportimport`) suggests a distractor.
5. **Model capacity. About +0.0005 to +0.001.**
   - Stage 2 trains on only 7.9M of about 24M non-validation pairs; use more, or iterate `qhash` slices and average.
   - Average 3 seeds.
   - For the final submission, retrain both stages including the validation fold (10% more data).
6. **France check.** France matches more on test: 63.5% of records get a parent against 58–60% elsewhere, and 4.9% of its S1 entities are empty against 5.5–5.7%.
   - If the prototype-2 leaderboard gap stays at about 0.007 while India/US are well calibrated, France may be over-merging.
   - Probe it: submit a variant that raises the France threshold only (seconds, from `work/test_prob.parquet`) and compare leaderboard scores.

Suggested order: **first resolve the train/test shift above** (upload the threshold probes). The ideas below are measured on validation, and prototype 2 showed that validation gains may not reach the leaderboard until the shift is understood.

## Machine notes
- 16 GB RAM, 12 cores, no GPU. Use `BER_JOBS=5`.
- The 2026-09-26 shutdown was not thermal (per the user).
- Stage-2 context over about 39M pairs peaks at about 10 GB of process memory. Don't run other heavy jobs alongside it.
- Every command needs `PYTHONIOENCODING=utf-8`.

## Rerun costs
| Rerun | Time |
|---|---|
| Stage 2 only (`train.py --stage2`) | about 35 min |
| Test prediction (`predict.py` + `postprocess.py`) | about 1 h |
| Features onward (`build_pairs` ×2, `train`, `predict`) | about 3 h |
| Full pipeline | about 7 h |
