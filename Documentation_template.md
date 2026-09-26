# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary
We treat the task as choosing a parent for each Source 2/3 record: in the training data every S2/S3 record matches at most one Source 1 entity. Candidates come from a country-partitioned sparse TF-IDF search with a combined name+address channel and an address-only channel; it captures 98.6% of true matches, and 98.3% survive pruning. Scoring has two LightGBM stages. Stage 1 scores each candidate pair from string similarities and *competition* features (how a candidate compares with the record's other candidates). Between the stages, *sibling expansion* adds candidates found through records already matched with confidence. Stage 2 then adds features built from stage-1 probabilities: competition in probability space, and *sibling* features that compare a record with the other records confidently assigned to the same entity. The key observation is that copies of one business share its typos, spacing and unit numbers. Finally, each S1 entity keeps the number of matches that maximises its expected F0.5. The held-out validation score is **0.9842 macro F0.5** (prototype 1: 0.9743).

---

## 2. Methodology

### 2.1 Problem Analysis
EDA findings on the training data:
- **Size.** Train has 2.21M S1, 5.03M S2 and 5.29M S3 records; test has 1.73M, 4.89M and 5.08M. Test adds France, which never appears in training.
- **One parent per record.** The ground truth holds 7.64M matched S2/S3 IDs, and every one is unique: no record matches two S1 entities. About 26% of S2/S3 records match nothing (distractors). 5.6% of S1 entities are singletons, and most have 2–6 matches.
- **Matches never cross countries,** so country is a safe blocking key. We still treat it as an open set of labels.
- **Name noise:** legal-suffix changes (Pvt/Private, Ltd/Limited, LLC/L.L.C.), word reordering, typos, bracketed or stray tokens (`[Corp]`, `>>`, `--`), accents (`Énterprises`), website forms (`zanderblue.com`), acronyms (`ZB`), DBA/AKA names (`Korbrixx D.B.A. Obsidian, LLC`), and names that are completely different while the address still matches.
- **Transliteration.** Indian names and states appear in nine Indic scripts: Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam, Oriya and Gurmukhi (`राम मार्केटिंग प्राइवेट लिमिटेड` = `Ram Marketing Private Limited`, `தமிழ்நாடு` = Tamil Nadu).
- **Address noise:** components reordered, typos (`ROCHESTEER`), abbreviations and wrong expansions (`Fremont SAINT` for Street), ordinals (`45ND`, `Fourth`), zero-padding (`005208`), number ranges (`1056-1060`), `null`/`N/A`/`<NULL>` placeholders, missing components, and about 3.3% fully empty. French records mix regions (`Hauts-de-France`) and departments (`Nord`), and use `N°`, `R.`, `All.`, `Bd`.
- **Hard negatives are common.** The same business name appears at different addresses. There are also near-miss distractors: records with no parent whose address matches an S1 entity except for one unit or house number (`Shop No. 315` vs `SHOP NO. 320`, `3751` vs `3760 Harrison Avenue`). These are the main source of false merges.
- **Siblings share an entity-level variant.** Within one S1 entity's matches, copies inherit the same rewritten address, the same typo (`BROWNS SUMMIT TOWWNSHIP` in two S2 records), the same zero-padding (`00882 Phillips Rd` in several S3 records) and even the same double space in the name. So a weak record often looks much more like a confidently matched sibling than like the S1 record itself. Most of our prototype-2 gain comes from exploiting this.

### 2.2 Solution Strategy
**Approach Type:** Multi-channel blocking + two-stage gradient-boosted pair classifier + sibling expansion + per-entity expected-F0.5 selection.
**Core Innovation:**
1. **Parent assignment.** Scoring runs from the S2/S3 side, so S1 candidates compete for each record, and the one-parent rule holds by construction.
2. **Sibling reasoning.** Stage-2 features compare a record with the records already confidently assigned to the same S1 entity. Sibling expansion uses the same idea to recover candidates that blocking missed.
3. **Stacking over stage-1 probabilities.** Competition among candidates is re-measured with learned probabilities instead of blocking scores, from both the record side and the S1 side.
4. **Learned transliteration and country-agnostic features.** A token-level Indic→Latin dictionary is learned only from training pairs. There is no country one-hot, so the model applies to France unchanged.

---

## 3. Candidate Generation (Blocking)
- **Normalisation.** Text is folded to lowercase ASCII (learned dictionary, then `unidecode`). Acronym dots are collapsed (`L.L.C.` → `llc`). Legal forms (including French SARL/SAS/EURL/SCI/SNC/EI…), street types and directions (including French: `R.` → rue, `Bd` → blvd, `All.` → allee, `PAS` → passage) and common name abbreviations (`Établissements`/`Ets`, `Compagnie`/`Cie`) are mapped to canonical forms. Ordinals are stripped, leading zeros removed, and digits split from letters. A comma component that is exactly a state is extracted as the state; for France, both region and department names map to one region code, so `…, Nord` and `…, Hauts-de-France` agree. Placeholder tokens (`null`, `N/A`, `N°`) are dropped.
- **Blocking keys (feature-hashed TF-IDF, 2^23 dims):**
  - **Name:** core words (legal forms and titles removed), consonant skeletons, word bigrams, the compact name or website domain, DBA/AKA alias words, numbers inside the name (`Local No 402`), and character 4-grams of the compact name (catches glued or split words such as `chiropracticcare`).
  - **Address:** normalised tokens including house numbers, consecutive token bigrams, 6-digit PIN codes, and consonant skeletons of address words (`cutlr`/`cutler`, `wakesha`/`waukesha`).
- **Search.** IDF is computed on Source 1 per country, and tokens found in more than 5,000 S1 records are dropped. The name and address halves are L2-normalised separately and concatenated, so the score is the mean of the name and address cosines. For every S2/S3 record, `sparse_dot_topn` returns the top 12 S1 entities of the same country by this score, plus the top 6 by address cosine alone. The address-only channel lets a record whose name is unrelated or written in another script still reach its parent. Candidates below 40% of the record's best score are dropped.
- **Final candidate set** (`candidate_pairs.tsv`): at most 8 candidates per record, each scoring at least 60% of the record's best, plus the 4 best by address cosine (≥ 0.4), plus the sibling-expansion candidates (Section 4).
- **Candidate pairs generated:** test has 91.0M raw, 38.6M after pruning and **39.9M final** (with 1.30M expansion pairs); train has 90.4M raw and 38.3M after pruning, plus 0.90M expansion pairs.
- **How we kept true matches from being lost.** Recall of true pairs on the validation fold:

| Stage | Prototype 1 | Prototype 2 |
|---|---|---|
| Raw blocking output | 97.74% | **98.58%** |
| After pruning | 97.38% | **98.33%** |
| + sibling expansion | – | recovers about half of the remaining misses in a validation experiment |

With a perfect classifier, the final candidate set would score 0.9947 macro F0.5.

---

## 4. Matching Model

**Stage 1 features (56):**
- **Name:** rapidfuzz `ratio`, `token_set_ratio`, `token_sort_ratio`, `partial_ratio` and `WRatio` on the normalised and core names; Jaro-Winkler and Levenshtein distance on the compact name; token-set and ratio on skeletons; domain-vs-name ratio and partial ratio; DBA-alias token-set; core-token overlap coefficient and intersection size; first-token similarity; token counts; a non-Latin-script flag; agreement of numbers inside the name.
- **Address:** token-set, token-sort, ratio and partial-token-set on the normalised address; street-word token-set, overlap and intersection; house-number set overlap and intersection; the number of house/unit numbers present on only one side, and the Jaccard of the two number sets (these target near-miss distractors); first house number equality, edit distance, and containment in the other's number list; PIN equality; state/region equality; an empty-address flag. Any comparison that is impossible because a field is missing is set to −1.
- **Blocking and competition:** combined, name and address cosines; rank and gap to the best candidate for the record; `margin` over the next-best candidate; number of candidates; the same rank and gap on the name and address cosines separately; from the S1 side, rank among the S1's queries, gap to its best query, number of queries, and how many queries rank this S1 first; the source (S2 or S3).

**Stage 1 models:** two LightGBM binary classifiers (MIT licence; 127 leaves, learning rate 0.15, early stopping). Each is trained on one half of the training records (split by a record hash; 7.8M pairs each, 24% positive). Every training pair is scored by the model that did not see it, so stage 2 trains on out-of-fold probabilities. Validation and test pairs get the mean of both models.

**Sibling expansion.** A record whose best stage-1 probability is below 0.5 looks up confidently matched records (probability ≥ 0.5) of the same country that share one of four keys:
- its normalised address;
- its core name;
- its name skeleton;
- its house number + street skeleton.

The S1 entities of those siblings become new candidates. Keys that point to more than 3 different entities are ignored as too generic. The new pairs get blocking cosines, context features and pair features like any other candidate, and are scored by stage 1.

**Stage 2 features (79):** all stage-1 features, plus:
- **Probability competition:** the stage-1 probability; its rank among the record's candidates; the best competing probability for the record; the sum over the record's candidates; the S1-side rank; the S1's best and summed probability over its other records; and how many other records choose this S1 confidently.
- **Sibling features:** these compare the record with the other records confidently assigned to the same S1:
  - the number of such siblings;
  - the best raw-string ratio (original spacing kept) of the name and of the address;
  - the best ratio and token-set ratio on normalised name and address;
  - whether the raw name or address is exactly equal to a sibling's;
  - equality of the first house number;
  - the best Jaccard of the house/unit number sets;
  - the same-source raw name and address ratios;
  - the number of expansion keys that produced the pair.

**Stage 2 model:** LightGBM, 63 leaves, learning rate 0.08, 821 rounds (early stopping), trained on 7.9M out-of-fold pairs of non-validation records.

**Decision rule:**
1. Each S2/S3 record goes to its most probable S1 candidate.
2. For each S1 entity, with p₁ ≥ p₂ ≥ … the probabilities of the records assigned to it, the top k records are kept. k maximises the expected F0.5, 1.25·(p₁+…+p_k) / (0.25·Σp + k).
3. No record is kept if P(no true match) = Π(1−pᵢ) is larger than that.

This replaces a global threshold and adds +0.0003 on validation.

**Validation design:**
- S1 entities are hashed into 10 folds, and fold 0 is held out.
- Validation takes every candidate of every S2/S3 record that touches a fold-0 entity, including expansion candidates, so the assignment step sees all competitors.
- Stage-2 training excludes all of these records.
- Stage-1 context features are computed over the whole candidate set, exactly as at test time.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9842** on the held-out validation fold (220k S1 entities, singletons included). Stage 1 alone: 0.9783; prototype 1: 0.9743.

| Stage-2 threshold | 0.30 | 0.50 | 0.60 | **0.66** | 0.75 | 0.90 |
|---|---|---|---|---|---|---|
| Macro F0.5 | 0.9799 | 0.9831 | 0.9837 | **0.9839** | 0.9837 | 0.9814 |

Expected-F0.5 selection per entity: **0.9842**.

| Change (validation) | Macro F0.5 |
|---|---|
| Prototype 1 (single LightGBM) | 0.9743 |
| + multi-channel blocking, French normalisation, number features, half-split stage 1, sibling expansion | 0.9783 |
| + stage 2 (probability competition + sibling features) | 0.9839 |
| + expected-F0.5 selection | **0.9842** |

- **Most important stage-2 features (gain):**
  - probability features: stage-1 probability, its rank for the record, the record's probability sum, `margin`;
  - `sib_nj` (unit-number agreement with siblings);
  - `ss_ra` (raw address similarity to a same-source sibling);
  - S1-side probability features;
  - `num_r_extra` (numbers present only in the record).
- **Common false positives (wrong merges):** records with no true parent that sit at the same building as an S1 entity but differ in one unit or house number, or carry a different name at the same address. The number-set and sibling number features reduced these, but they remain the main source of precision loss.
- **Common false negatives (missed matches):**
  - **Empty-address records.** 61% of the remaining missed true pairs have an empty address. When the name fits two S1 entities in different cities (`Carley's Great Wireless` in MD and IN), the record carries no evidence to decide between them.
  - **Unrelated names at the right address.** An invented name (`veogild`, `korhalovantage`) at an address that matches the S1 entity; about 8–12% of misses.
  - **Noisy house numbers.** A true copy whose house number changed through noise (`2761` vs `2760`) can look like a near-miss distractor.
- **Generalising to France:** France has no training labels. The region/department mapping and French abbreviation rules make its address features behave like US and Indian ones. On test, France matches slightly more than the other countries: 63.5% of its S2/S3 records get a parent (India 58.2%, US 59.9%), and 4.9% of its S1 entities are left empty (India 5.7%, US 5.5%). Without labels we cannot tell whether France simply has more true matches or whether the model over-merges there; the leaderboard is the check.

---

## 6. Conclusion
Treating the task as parent assignment gives a clean, fully CPU-based pipeline. Robust normalisation, multi-channel blocking and two-stage scoring together take it to 0.984 macro F0.5 on held-out data. Two things mattered most:
- measuring where true pairs were lost (blocking, threshold, or the wrong parent) before changing the model;
- recognising that the copies of one business share an entity-level variant, and using it both as features (sibling similarity) and for recall (sibling expansion).

The remaining headroom is concentrated in empty-address records that fit several same-name entities, and in near-miss distractors. Both call for reasoning over whole clusters of records rather than individual pairs.

---

## Appendix

### A. Code Artefacts
The code is in `code/business_entity_resolution/`. All source is under `src/`, with `README.md` and `requirements.txt` (Python 3.13; numpy, pandas, scipy, scikit-learn, pyarrow, rapidfuzz, lightgbm, sparse_dot_topn, Unidecode). The entry point is `python src/run_all.py`, which runs:

| Step | Script | Output |
|---|---|---|
| Learn transliteration dictionary | `learn_translit.py` | `work/translit.json` |
| Normalise records | `prep.py train/test` | `work/<split>_source{1,2,3}.parquet` |
| Blocking | `block.py train/test` | `work/<split>_cand_raw.parquet` |
| Pruned candidates + features | `build_pairs.py train/test` | `work/<split>_cand_final.parquet`, `work/<split>_pairs.parquet` |
| Stage 1, sibling expansion, stage 2, threshold | `train.py` (uses `expand.py`, `stack.py`) | `work/model_s1_{0,1}.txt`, `work/model_s2.txt`, `work/threshold.json` |
| Predict | `predict.py` | `output/matching_results.tsv`, `output/candidate_pairs.tsv`, `work/test_prob.parquet` |
| Per-entity match selection | `postprocess.py` | `output/matching_results.tsv` |

A full rerun takes about 7 hours on a 12-core, 16 GB, CPU-only Windows machine with `BER_JOBS=5`. Blocking takes about 3 hours of that. Peak memory is about 10 GB, reached while building stage-2 context over the 39M test pairs. Blocking, feature building and prediction otherwise work on one country or source at a time and stream to disk.

### B. Additional Results
- **Blocking channels**, measured on a sample of 20k true-matched records per country. The final configuration recovers about a third of the pairs the prototype-1 blocking missed.

| Configuration | India recall | US recall | Pairs per record |
|---|---|---|---|
| Prototype 1 tokens | 96.77% | 98.39% | 6.1 / 4.9 |
| + address skeletons + name char 3-grams | 97.27% | 98.72% | 6.2 / 5.0 |
| + address-only top-6 channel | 97.68% | 98.97% | 8.5 / 7.7 |
| **+ char 4-grams instead of 3-grams (final)** | **97.77%** | **99.02%** | 8.6 / 7.8 |

- **Ideas tested and dropped:**
  - A third stage that recomputes sibling features from stage-2 probabilities: −0.001.
  - Sibling features measured relative to the record's competing candidates: +0.0001, not kept.
- **Fair play:** no external databases, APIs, geocoding or pretrained models were used. US and Indian state abbreviations, French region and department names, and street-type abbreviations are hand-written normalisation rules. The Indic transliteration dictionary is learned only from the provided training labels.
