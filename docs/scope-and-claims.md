# Scope and Claims

This document states exactly what PathoPress can and cannot claim from the evidence
currently in the repository. It is written to be adversarially readable: every number is
traceable to a measured artifact, and every claim carries the scope conditions under
which it holds. Where a number is a provisional upper bound rather than a defensible
estimate, it is labelled as such inline.

## 1. What the project is

PathoPress ports the BenchPress idea — that a large benchmark matrix is highly
redundant, so a small selected subset of benchmarks can stand in for the whole — from
LLM evaluation to computational pathology foundation models.

The substrate is a matrix of **59 pathology foundation models x 187 evaluation
protocols**, with **2,122 observed cells** (19.2% density after filtering). Observations
are sparse and unevenly distributed: the median evaluation has **7 observed models**
(min 5, q25 7, q75 8, max 32). Coverage comes from six sources:

| Suite | Evaluations |
| --- | --- |
| Patho-Bench | 122 |
| HEST | 18 |
| THUNDER | 16 |
| EVA | 15 |
| H-Optimus-1 report | 10 |
| PathoROB | 6 |
| **Total** | **187** |

The machinery is a low-rank matrix-completion model over this matrix
(`src/pathopress/completion.py:163`, rank-1 bias-ALS fast path at
`src/pathopress/completion.py:73`), a greedy forward probe selector over evaluations
(`src/pathopress/probes.py:832`), and a pairwise ranking-preservation evaluator
(`src/pathopress/ranking.py:134`). The intended product question is: *if you can only
run a handful of pathology evaluations on a new model, which ones should you run, and
how well can you recover the rest?*

## 2. The scale problem: why raw MedAE misleads across domains

The most tempting cross-domain claim — "PathoPress reconstructs pathology scores about
three times more accurately than BenchPress reconstructs LLM scores" — is an artifact of
the two domains having different score dispersions. It should not be made.

Measured on the upstream BenchPress matrix (84 models x 133 benchmarks, 2,604 cells)
against the PathoPress matrix. **The upstream matrix is not vendored in this repository.**
It is an external dependency: reproducing the BenchPress column of the table below
requires a separate checkout of `microsoft/benchpress` at pinned commit
`0a684b63ee0e4a401cb907a3827a82ea997d74c4`. Every BenchPress-side dispersion figure here
(84 x 133, 2,604 cells, median SD 14.1, MedAE 4.6, and the 0.33 ratio derived from them)
is therefore **not reproducible from this repository alone**.

| Per-column dispersion | BenchPress (LLM) | PathoPress (pathology) |
| --- | --- | --- |
| Median SD | 14.1 | 3.75 |
| Median IQR | 15.8 | 3.80 |
| Median range | 46.7 | 10.60 |
| Columns with SD < 5 | ~4% | 69% |

Pathology benchmark columns are roughly four times narrower than LLM benchmark columns.
Any absolute-error metric inherits that compression directly, so a smaller MedAE in
pathology is largely a restatement of the fact that pathology models are closer together,
not evidence of a better method.

Dispersion is also strongly metric-dependent *within* PathoPress, which means a single
pooled error figure hides most of the variation:

Population: median per-column SD over the **59-model x 187-evaluation analysis matrix**
(`experiments/analysis_matrix.npz`), restricted to columns with at least two observed
models.

| Column metric | Median SD |
| --- | --- |
| `pearson_r` | 1.16 |
| `auroc` | 1.36 |
| `balanced_accuracy` | 6.09 |
| `robustness_index` | 9.91 |

The `robustness_index` figure was previously given as 10.43. That value is the median SD
over all 62 models in `data/scores.csv`, a different and larger population than the other
three rows, which were already matrix figures. On the matrix the value is 9.9074, shown
here as 9.91.

Footnote on span: these four metrics are not the full set. Within the matrix,
`clustering_score` has the highest median SD at **10.2849**, above `robustness_index`. Any
"SDs span 1.16 to 9.91" statement therefore describes only the four metrics tabulated
above, not every metric in the matrix.

The consequence for ranking is direct: the median true score gap between two models on a
given evaluation is **3.10 normalized points**, and **64.8% of model pairs differ by less
than 5 points**. Most of the ranking problem is therefore made of near-ties, which is
exactly the population that absolute margin thresholds discard (see 3.1).

The scale-free comparison is the ratio of reconstruction error to native column
dispersion:

- BenchPress: 4.6 / 14.1 = **0.33**
- PathoPress: 1.609 / 3.75 = **0.43**

**On the corrected scale, the pathology port is modestly *worse* than the upstream LLM
result, not better.** The raw MedAE advantage (1.609 vs 4.6) is a scale artifact and must
not be reported as a method improvement. This is the single most important claim
correction in this document.

## 3. Supported claims

### 3.1 Centerpiece: greedy probe selection beats random probe selection at every margin

This is the strongest result the project has, and it is strong specifically because it is
**margin-invariant**. A margin threshold discards model pairs whose true scores are closer
than the margin; sweeping the margin therefore sweeps how easy the ranking task is. If an
advantage only appeared at some thresholds, it would be a threshold artifact. It does not.

Configuration: greedy k = 10 probes, rank-1 completion, `any_candidate` selection, all-known
track; the random baseline is averaged over 10 repeats. Sweep driver
`experiments/run_ranking_preservation.py`. `median_accuracy` equally weights columns;
`pooled_accuracy` weights by pair count. Predicted ties count as errors
(`src/pathopress/ranking.py:134`).

| Margin type | Margin | Pairs retained | Columns | Greedy median | Greedy pooled | Random median | Random pooled |
| --- | --- | --- | --- | --- | --- | --- | --- |
| absolute | 0.0 | 17,159 | 187 | 0.679 | 0.771 | 0.552 | 0.607 |
| absolute | 1.0 | 13,763 | 187 | 0.708 | 0.811 | 0.567 | 0.623 |
| absolute | 2.0 | 10,973 | 178 | 0.746 | 0.836 | 0.571 | 0.633 |
| absolute | 3.0 | 8,845 | 168 | 0.785 | 0.849 | 0.585 | 0.638 |
| absolute | 5.0 | 6,048 | 148 | 0.878 | 0.861 | 0.603 | 0.647 |
| absolute | 10.0 | 2,722 | 100 | 1.000 | 0.883 | 0.663 | 0.657 |
| relative | 0.25 x SD | 14,397 | 187 | 0.696 | 0.804 | 0.559 | 0.619 |
| relative | 0.25 x IQR | 13,657 | 187 | 0.696 | 0.810 | 0.562 | 0.622 |
| relative | 0.50 x SD | 11,957 | 187 | 0.712 | 0.829 | 0.567 | 0.629 |
| relative | 0.50 x IQR | 10,751 | 187 | 0.712 | 0.838 | 0.575 | 0.635 |
| relative | 1.00 x SD | 8,232 | 187 | 0.750 | 0.862 | 0.567 | 0.642 |
| relative | 1.00 x IQR | 6,076 | 187 | 0.778 | 0.863 | 0.592 | 0.655 |

Greedy exceeds random on all four accuracy columns at all twelve settings, under both
absolute margins and margins defined relative to each column's own SD or IQR. The relative
rows matter most: because they scale the threshold to each column's dispersion, they retain
all 187 columns and cannot be dismissed as a consequence of dropping hard columns.

**The honest unconditional number is margin 0: greedy 0.679 median pairwise accuracy
versus random 0.552** (pooled: 0.771 vs 0.607), over all 17,159 pairs in all 187 columns.

**The previously headlined 0.878 is the most flattering point on the curve and should not
be the headline.** It is the greedy median at absolute margin 5, which retains 6,048 of
17,159 pairs — **35% of pairs**, and only 148 of 187 columns. The 1.000 at margin 10 is
worse still: it retains 2,722 pairs (16%) across 100 columns. Both figures are reachable
only by discarding the pairs the method finds hardest, and section 2 shows that 64.8% of
all pairs sit inside the 5-point margin, so margin 5 removes the majority of the real
decision problem. Wherever 0.878 appears it must be restated with its margin and
pair-retention alongside the margin-0 number. That restatement has already been made at
every current occurrence: `docs/benchpress-parity.md` (prose at the margin-5 discussion,
the sweep table row, and the "most favorable point on the curve" note) and
`docs/imputation.md` (margin-5 prose in the ranking-validation section). `README.md`
does not quote 0.878 at all — the earlier pointer to `README.md:47` was wrong, as were the
specific line numbers given for the other two files.

Scope conditions on this claim: k = 10, rank 1, all-known regime; greedy selection is
in-sample (see 5.3); the random baseline used here is the in-sample random-probe
comparator. The held-out k = 0 and random-probe controls have now been run for *MedAE*
under the matched-cell LOFO protocol (see 3.5); the held-out *ranking* control remains
unrun (see 6).

### 3.2 Cell-level reconstruction accuracy

Rank-1 completion achieves **pooled MAE 3.135 and MedAE 1.609** over 21,181 repeated
held-out prediction instances drawn from the 2,122 unique observed cells (10 seeds x 3
folds). This claim stands on its own terms and is unaffected by the corrections elsewhere
in this document. It must be reported with the dispersion context of section 2: relative to
a median column SD of 3.75, a MedAE of 1.609 is a ratio of 0.43, which is not better than
upstream BenchPress.

Scope condition: this is a **within-model** cross-validation. All 59 models remain visible
in every fold; only cells are held out. It measures filling gaps for models already in the
matrix, not cold-start prediction for an unseen model.

### 3.3 Temporal deployment on 2025-release models

For N = 7 models released in 2025, selected by a hard date rule with prediction errors not
inspected during selection, per-target MedAE under increasing probe budgets:

| Model | k = 1 | k = 5 | k = 10 |
| --- | --- | --- | --- |
| exaone-path-2.5-slide | 2.019 | 2.110 | 1.514 |
| h-optimus-1 | 0.970 | 0.796 | 0.537 |
| h0-mini | 1.252 | 0.758 | 0.502 |
| midnight | 1.317 | 0.962 | 0.608 |
| openmidnight | 0.802 | 0.507 | 0.160 |
| threads-slide | 1.935 | 1.527 | 1.049 |
| uni2-h | 1.868 | 1.557 | 1.429 |

Supported claim: **error decreases with probe budget for every one of the 7 models from
k = 1 to k = 10.** The date rule makes this a genuine forward-looking test rather than a
retrospective split. Two scope conditions: the progression is not uniformly monotone —
`exaone-path-2.5-slide` worsens from k = 1 (2.019) to k = 5 (2.110) before improving at
k = 10 — and N = 7 is far too small to support any claim about a rate, a scaling law, or a
recommended budget. Per-model spread at k = 10 is wide (0.160 to 1.514), so no single
expected error should be quoted for a new model from this table.

### 3.4 Interval coverage for unseen models

New-model prediction intervals achieve **94.8% empirical coverage against a nominal 90%
target**. Two disclosures are mandatory whenever this number is quoted. First, coverage is
computed over the **calibrated subset only**: 31,163 of 33,272 targets, with **2,109
abstentions** where no interval was issued. Coverage on the issued intervals says nothing
about the abstained 6.3%. Second, 94.8% against a 90% target is over-coverage, meaning the
intervals are **conservative** — wider than the nominal level requires. They are safe to
quote as "at least 90%" and must not be described as well-calibrated.

### 3.5 LOFO held-out model scorecard reconstruction, on matched cells

Under a leave-one-family-out (LOFO) protocol, all 59 models are held out exactly once
across 34 family folds. Median validation set size is 1 model per fold (min 1, max 7);
58 training models per fold at the median. The LOFO design is statistically correct:
no held-out model appears in the training set for its own fold.

**The arms must be scored on the same cells.** An earlier version of this comparison did
not do that: the k=0 arm was scored on every observed validation cell, while the greedy
and random arms were each scored on whatever remained after removing their *own* revealed
probe cells. Three arms on three different denominators are not comparable, and the
resulting "31.2% error reduction" (greedy 1.7994, k=0 2.6143, random 2.6078) is withdrawn.

The corrected protocol fixes one cell set per fold and per depth k: it excludes the union
of the cells revealed by the greedy prefix and by all 10 random repeats, then scores every
arm on the identical remainder. At k=4 this excludes 486 of 2,122 cells and scores all
arms on the 1,636 matched cells.

| Arm at k=4 (matched cells) | MedAE | Aggregation convention |
| --- | --- | --- |
| greedy | **1.8781** | median of 34 fold medians |
| k=0 baseline | **2.6524** | median of 34 fold medians |
| random control | **2.6013** | **convention A**: median over all 340 fold × repeat MedAEs |
| random control | **2.6260** | **convention B**: median of the 34 per-fold medians over repeats |

The point estimate of the reduction versus k=0 is 29.2%. **That figure must not be quoted
to three significant figures.** Bootstrapping over folds gives a 95% CI of
**[2.8%, 58.7%]** against k=0 and **[3.4%, 53.5%]** against random — an interval spanning
an order of magnitude. The magnitude of the improvement is not estimable at useful
precision from 34 folds.

What *is* solid is the paired per-fold comparison, and that should lead:

- greedy beats the k=0 baseline in **18 of 34 folds**, Wilcoxon signed-rank
  **p = 0.0088**;
- greedy beats the random control in **22 of 34 folds**, Wilcoxon signed-rank
  **p = 0.0151**.

**Convention mismatch, stated explicitly.** The paired test against the random arm is
computed on **convention B** (fold medians, 2.6260) because a paired test needs one random
value per fold, while the headline row most often quoted for the random arm is
**convention A** (2.6013). The table value and the test statistic therefore do not use the
same aggregation, and neither does the bootstrap CI against random ([3.4%, 53.5%]), which
is also convention B. This is disclosed rather than harmonised because both aggregations
are defensible; it must never be left implicit. The greedy and k=0 arms have only one
value per fold, so no such ambiguity applies to them or to the 18/34, p = 0.0088 test.

So the supported claim is that greedy probe selection reduces held-out reconstruction
error relative to both controls, with a directional result that survives a paired test,
and an effect size that is imprecise. This remains the primary out-of-sample evidence for
the probe-selection method.

Random-arm aggregation convention: the random arm has two defensible aggregations and
they differ at every k. **Convention A** — the median over all 340 fold × repeat MedAEs —
gives **2.6013**. **Convention B** — the median of the 34 per-fold medians over repeats —
gives **2.6260**. Any quoted random-arm number must name its convention, and any published
pairing of a random-arm value with a significance test or CI must state which convention
each of the two uses (see the convention-mismatch note above: the tests and CIs are
convention B).

Reproduction: `scripts/replay_lofo_matched_cells.py`, artifact
`experiments/lofo_matched_cells_rank1.json`.

### 3.6 Per-evaluation utility: a null result

**The previously published claim that 58.9% of columns show positive utility is
withdrawn.** It was produced by a broken metric. The numerator was `parity_medae`, a
matrix-wide quantity that spans only 1.750–2.646 across all columns; the denominator was a
column-scoped leave-one-out baseline spanning 0.150–32.1, a 214× range. Dividing a nearly
constant numerator by a wildly varying denominator makes the resulting skill score close
to a monotone re-encoding of each column's own dispersion, not a measure of whether probe
selection helped that column.

The corrected measurement is per-column, leave-one-out, and on matched cells. The
positivity rule is stated explicitly: **a column counts positive when its matched greedy
k=4 MedAE is below its matched k=0 MedAE for that fold.**

**86 of 174 scored columns (49.4%, bootstrap 95% CI [42.0%, 56.9%]) are positive.** Twelve
columns are excluded by the noise floor and one has no matched cells, leaving 174 of 187
scored.

**This is indistinguishable from a coin flip. It is not a positive result and must not be
reported as one.** The CI contains 50%.

The rule is not incidental: an alternative rule scoring each column against its
leave-one-out column-median baseline also totals 86/174, but disagrees with the rule above
on 56 individual columns. Two rules that agree on the total while disagreeing on a third
of the columns are measuring something noisy, which is consistent with a null.

The suite breakdown **inverts** relative to what was previously published. The old table
was an artifact of the denominator, not a finding about which suites benefit:

| Suite | Old (withdrawn) | Corrected |
| --- | --- | --- |
| PathoROB | 6 / 6 | 2 / 6 |
| THUNDER | 13 / 16 | 14 / 16 |
| Patho-Bench | 77 / 122 | 45 / 116 |
| EVA | 5 / 15 | 9 / 15 |
| H-Optimus-1 report | 2 / 10 | 4 / 7 |
| HEST | 0 / 18 | 12 / 14 |

The corrected denominators are the scored columns per suite after noise-floor and
no-matched-cell exclusions. **The previous statement that the method fails outright on
HEST and on spatial-transcriptomics regression is withdrawn**; it was a consequence of the
denominator, and on the corrected measurement HEST is 12/14 positive. No suite-level claim
in either direction should be drawn from these counts.

Noise-floor disclosure. Including all 187 columns gives **94 of 187 positive (50.3%, CI
[43.3%, 57.2%])** — the same null. **Eight of the 12 noise-floor-excluded columns are
positive.** The excluded columns have leave-one-out baselines in the range 0.15–0.95
against a median of about 2.50 among included columns, so the exclusion
(`SKILL_NOISE_FLOOR_DISPERSION = 0.5`) is confounded with structurally-near-guaranteed-
negative status rather than being an independent filter. The exclusion is disclosed
because it is not neutral, but it does not change the conclusion.

Reproduction: `scripts/replay_lofo_matched_cells.py`, artifact
`experiments/lofo_matched_cells_rank1.json`.

### 3.7 The 25-task allowlist arm does not support a selection claim

The 25-task pipeline-feasibility allowlist was proposed as a restricted candidate pool.
On its own matched cell set (1,581 cells at k=4), allowlist greedy reaches MedAE
**1.9951**, against **1.7234** for greedy over any candidate on the same cells. Restricting
the candidate pool makes reconstruction worse, as expected.

More importantly, within the allowlist there is no selection signal that survives a
significance test. Against a genuine allowlist-restricted random control on the arms'
shared matched-cell set at k=4 (**1,237 cells**, 34 LOFO folds), allowlist greedy reaches
**1.9463** and allowlist random **2.0251** under the median-of-fold-medians convention
(**2.0118** under the median-over-340-fold-×-repeat convention); the k=0 arm on the same
cells is **2.8555**. Greedy is nominally ahead, but wins only **10 of 34 folds**, ties 15,
and loses 9; the Wilcoxon signed-rank test gives **p = 0.4939** and the bootstrap CI on
the reduction is **[-11.5%, +20.7%]**. The picture does not change with depth: across
k = 1..5 the p-values range from 0.05 to 0.49 and every CI straddles zero.

**Mandatory disclosure: this test is underpowered by design.** The 15 ties are exactly
the **15 of 34 folds** where the allowlist arms are *zero-information* — the held-out
family has no observed score on **any** of the 25 allowlist evaluations, so both arms
produce literally identical predictions (not merely "too close to call") and collapse
onto the k=0 arm. This identity holds at every k = 1..5. Restricting to the 19
informative folds alone (10 wins vs 9 losses) is still non-significant, so the null
is not an artifact of tie-handling. Only 19 folds contribute any signal at all. The
correct reading is therefore "no advantage demonstrated within a pool that is too sparse
to test", not "greedy is worse than random". For calibration, the unrestricted
187-candidate arm on the same protocol beats random in 22 of 34 folds at p = 0.0151.

**This is a negative result and is reported as one.** Nothing here supports a claim that
useful probes can be chosen from within the 25-task feasibility pool.

Note that the previously published pair (allowlist greedy 2.0404 versus allowlist random
2.0109, "greedy worse under both conventions") is withdrawn: neither value occurs in any
artifact, and the sign of the comparison was inverted.

## 4. Claims explicitly not supported

### 4.1 "PathoPress reconstructs pathology scores ~3x more accurately than BenchPress"

Not supported. The MedAE gap (1.609 vs 4.6) tracks the four-fold difference in column
dispersion. On the error-to-dispersion ratio the port is worse (0.43 vs 0.33). See
section 2.

### 4.2 "Only THUNDER evaluations benefit" / "utility is 4.8%"

Not supported as a scientific finding — it is a baseline artifact. The original utility
metric was `improvement_over_column_median = 1.9000 - parity_medae`, which compares every
column against a **single global pooled constant**. Under that metric only 9 of 187
evaluations (4.8%) showed positive utility, and all 9 were THUNDER. Because THUNDER columns
happen to sit near the global constant, this metric measured proximity to a pooled number
rather than model-discriminative signal. It must not be reported. Its replacement is
section 3.6, which uses the matched-cell LOFO protocol and reports 86/174 columns (49.4%)
positive — a null result. Note that the intermediate replacement metric, which reported
103/175 columns (58.9%), was itself broken for a different reason and is also withdrawn;
see 3.6.

### 4.3 "PathoPress identifies a cheap subset of evaluations"

Not supported at all. **Zero of 187 evaluations carry cost data** — no runtime, no hardware,
no annotation hours, no dollar figures. The BenchPress "cheap subset" framing has no
empirical support in this repository. The 25-task "low-friction" pool is a **sample-count
feasibility proxy**, not a cost model, and must always be described that way
(`docs/budgeted-probe-selection.md` already states this; `docs/evaluation-cost-evidence.md`
records the negative result).

### 4.4 "Held-out ranking accuracy is 1.0"

Not supported. Under the group-wise held-out split, the reported
`median_accuracy = 1.0` is a **counting artifact**: 57 of the 75 eligible columns (under
the `>= 5` normalized-point gap rule; 56 of 73 under a strict `> 5` gap) contain
exactly one comparable model pair, so the per-column accuracy is forced to 0 or 1 and the
median collapses to 1.0. It is not evidence of perfect ranking. See 5.1.

### 4.5 "The LOFO held-out ranking result is 0.804"

Not supported as a precise estimate. The LOFO protocol is the statistically correct design,
but at N=59 with 34 family groups, 19 folds have exactly one validation model, so most
folds contribute zero or one comparable pair. The `pairwise_n_pairs` value at k=10 is 1,
meaning the 0.804 figure is a mean across per-fold metrics computed on ~1-2 pairs each.
It is not estimable at useful precision and should not be quoted as an improvement over
the prior single-split value of 0.775 — the protocols are not directly comparable. The
held-out *MedAE* result (3.5) is well-supported; it is specifically the held-out *ranking*
number that is under-supported.

### 4.6 "Greedy beats random on held-out data"

Not yet fully supported for held-out ranking. The in-sample all-known greedy-versus-random
comparison (3.1) is strong and margin-invariant. The held-out MedAE result (3.5) shows
greedy beating both the random control and the k=0 baseline on matched cells, in 22/34 and
18/34 folds respectively (Wilcoxon p = 0.0151 and p = 0.0088); the size of that reduction
is imprecise, with a bootstrap CI of [3.4%, 53.5%] versus random and [2.8%, 58.7%] versus
k=0. However, the held-out *ranking* result is under-supported due to sample size — at
N=59 with 34 family groups, most folds contribute zero or one comparable pair (see 4.5).
The in-sample all-known result remains the headline ranking claim.

### 4.7 Any claim of clinical utility

Not supported and not attempted. Nothing in this repository evaluates downstream clinical
outcomes.

## 5. Threats to validity

### 5.1 Model non-independence

The 59 models are not independent draws. **40 of 59 (67.8%) belong to a multi-model
family**, across 34 family groups (32 non-empty plus 2 blank-family singletons). The
largest families are Kaiko (7), DINOv2 (4), DINOv3 (3), H-Optimus (3), and Midnight (3),
plus **10** two-model families. The arithmetic is self-consistent: 15 multi-model families (10 two-model families plus 5 larger)
account for 40 models, 17 non-empty singleton families and 2 blank-family singletons
account for the remaining 19, and 15 + 17 + 2 = 34 groups. This is confirmed by the LOFO fold
structure in `experiments/probe_selection_results_rank1.json` (34 folds: 15 with 2+ validation
models, 19 with exactly 1). (The previously published
"42 of 59 (71.2%)" and "11 two-model families" are withdrawn; both contradict
`data/model_metadata.csv` and the 34-fold structure in
`experiments/lofo_matched_cells_rank1.json`.) Random model splits therefore leak
architecture and
pretraining-corpus information across the split, and any accuracy measured under a random
split is optimistic.

A group-wise split fixes the leak but shrinks the evaluable set severely. Under a 70/30
group-wise split at seed 42 the validation side is **11 of 59 models (18.6%), all of them
singletons, with every flagship family entirely in train** — so the split does not test
generalization to a new family so much as it tests the leftovers. The result is also
unstable across seeds: n_val swings 11 / 17 / 18 / 19 / 20 for seeds 42 / 0 / 7 / 1 / 123.
At seed 42 this leaves only 334 observed validation cells, only **22 of 187 columns** with
3 or more validation models, and — under the ranking evaluator's documented margin rule,
which counts a validation model pair as comparable when their true normalized-score gap is
**>= 5** points — **75 eligible columns of which 57 contain exactly one comparable pair**,
the direct cause of the 1.0 artifact in 4.4. The rule must be stated because it moves the
count: under a strict `> 5` gap the same recomputation gives 73 eligible columns and 56
single-pair columns. The published 57 corresponds to the `>= 5` rule. (The previously
quoted "74 eligible columns" matches neither rule and is withdrawn.) The honest summary is that
the group-wise design is the correct one and the current matrix is too small to support it.

### 5.2 Within-model cross-validation

The headline cell-level result (3.2) holds out cells, not models. Every model is visible in
every fold. This measures interpolation into a partially observed row, not cold-start
prediction. Cold-start behaviour is addressed only by the N = 7 temporal set (3.3) and the
new-model intervals (3.4).

### 5.3 In-sample rank and probe selection

Rank selection was performed using the same folds used for reporting; there is no nested
cross-validation. Greedy probe selection is likewise fit on the matrix it is evaluated
against. Both inflate reported performance by an unquantified amount. The margin-invariance
of 3.1 argues the greedy-over-random *ordering* is robust, but not that the *magnitudes*
are unbiased.

### 5.3a The greedy selector optimizes a partly self-serving objective

Greedy selection scores candidate probes with `parity.median_absolute_error`, which
includes the revealed probe cells and scores them as literal 0.0. A probe that would
otherwise have been predicted badly therefore improves the objective simply by being
revealed, independent of how much it tells the model about the rest of the row. The
selector is thus partly optimizing "reveal the cells that would be predicted worst"
rather than pure informativeness.

The gap is measurable. At k=4 the selection objective reads **1.5142** against the
held-out quantity **1.7994** — the objective is about **15.9% optimistic** relative to
what it is being used to improve.

This is disclosed as a limitation, not corrected. Fixing it means selecting on a
hidden-only objective, which would change which probes are chosen and requires an
approximately 8.7-hour rerun of the selection stage. Until that rerun happens, the probe
sets reported here are the ones this biased objective chose. The held-out matched-cell
results in 3.5 are still scored on cells no arm revealed, so the bias affects *which
probes were selected*, not the honesty of the reported held-out numbers.

### 5.4 Small and uneven N

Median 7 observed models per evaluation (min 5) means most columns support very few pairwise
comparisons. The 19.2% density means the completion model is extrapolating over most of the
grid. Suite-level per-evaluation utility counts (3.6) rest on small scored denominators —
H-Optimus-1 report contributes 7 scored columns, PathoROB 6, HEST 14 — and the counts
inverted entirely when the metric was corrected. Suite-level conclusions from single-digit
or low-double-digit counts are not reliable in either direction.

### 5.5 Missing cost data

With 0/187 evaluations carrying cost data, the core BenchPress economic argument — that a
selected subset saves measurable compute or money — cannot be tested here at all. This is a
gap in the evidence, not a negative result about the method.

### 5.6 Metric heterogeneity

Columns mix `pearson_r`, `auroc`, `balanced_accuracy`, and `robustness_index`, whose median
per-column SDs over the 59 x 187 analysis matrix span 1.16 to 9.91 (section 2; the span
covers those four tabulated metrics only — `clustering_score` is higher still at 10.28).
Pooled error and pooled accuracy figures are dominated
by the high-dispersion metrics. Per-metric breakdowns should accompany any pooled figure.

### 5.7 Under-supported held-out ranking estimate

The LOFO protocol is the statistically correct design for held-out evaluation, but at N=59
with 34 family groups it leaves too few independent units to estimate held-out *ranking*
accuracy with useful precision. Nineteen of 34 folds have exactly one validation model, so
most folds contribute zero or one comparable pair. At k=10, `pairwise_n_pairs` is 1, meaning
the mode-level held-out ranking figure (0.804) is a mean over per-fold metrics computed on
~1-2 pairs each. The 0.775 value from the prior single-split is not a baseline the new
number can meaningfully improve on — the protocols use different model splits, different
training set sizes, and different pair populations. Do not quote 0.804 without `n_pairs`
alongside it; do not describe it as an improvement. The held-out *MedAE* result is
well-supported (3.5); the held-out *ranking* result is not.

## 6. Open work required before publication

1. **State the held-out ranking limitation wherever 0.804 appears.** Quote it only with
   `n_pairs` alongside it, or describe it as not estimable at useful precision. The LOFO
   protocol is the statistically correct design but leaves too few independent pairs. See 4.5
   and 5.7.
2. **Run the held-out k = 0 random-probe control for ranking.** The in-sample
   greedy-versus-random comparison (3.1) is strong; the held-out MedAE controls are now
   run on matched cells (3.5). The held-out *ranking* control would be needed to close 4.6.
3. ~~**Recompute per-evaluation utility with a leave-one-out column baseline and a MAD
   denominator.**~~ **Done.** This has been run. The result is in 3.6: the corrected
   per-column, leave-one-out, matched-cell measurement is 86/174 (49.4%, CI
   [42.0%, 56.9%]), which is null. The 58.9% figure it replaced is withdrawn.
3a. **Rerun greedy selection against a hidden-only objective.** The current selector
   optimizes an objective that is about 15.9% optimistic at k=4 because it scores revealed
   probe cells as zero (5.3a). Fixing this changes which probes are selected and needs an
   approximately 8.7-hour rerun.
4. ~~**Restate 0.878 with scope wherever it appears.**~~ **Done.** Every current
   occurrence — in `docs/benchpress-parity.md` and `docs/imputation.md` — now pairs it
   with margin 5, 35% pair retention, 148/187 columns, and the margin-0 pair
   0.679 vs 0.552. `README.md` does not quote the figure.
5. **Replace all raw cross-domain MedAE comparisons with the error-to-dispersion ratio**
   (0.43 vs 0.33), and state explicitly that the port is modestly worse on that scale.
6. **Retire the `improvement_over_column_median = 1.9000 - parity_medae` metric** and the
   "only THUNDER benefits" narrative built on it.
7. **Withdraw the held-out `median_accuracy = 1.0` claim** or report it with the
   57-of-75 single-pair column count that produces it (`>= 5` gap rule; 56-of-73 under a
   strict `> 5` gap — state the rule alongside the count).
8. **Add nested cross-validation for rank selection**, or report the reported figures as
   optimistic with an explicit statement to that effect.
9. **Report group-wise split results across all five seeds**, not seed 42 alone, given the
   11-to-20 swing in validation size (5.1).
10. **Collect cost data**, or permanently drop the "cheap subset" framing and present the
    work as redundancy compression only.
11. **Expand the matrix toward more independent model groups**, since 5.1 is a sample-size
     problem that no reanalysis can fix — and is the root cause of 5.7.

## 7. Reproduction

All numbers in this document derive from generated artifacts under `experiments/` and
`outputs/`, which are not committed in usable form. **The tables here will not refresh on
their own — the artifacts must be regenerated first**, and any figure re-derived after
regeneration should be checked against this document before it is quoted.

Regeneration path:

1. Rebuild the matrix and fold substrate: `scripts/build_shared_artifacts.py`.
2. Rebuild the score ledger and verify it: `scripts/build_score_review_ledger.py`, then
   `scripts/validate_score_review_ledger.py`.
3. Regenerate the margin sweep in 3.1: `experiments/run_ranking_preservation.py`
   (k = 10, rank 1, all-known).
4. Regenerate cell-level CV (3.2): `experiments/run_method_comparison.py`
   (`--prepare-folds` then `--merge`).
5. Regenerate confidence coverage (3.4): `experiments/run_new_model_confidence.py`.
6. Regenerate the temporal table (3.3): `experiments/run_temporal_deployment.py`.
7. Regenerate the matched-cell LOFO results (3.5, 3.6, 3.7):
   `scripts/replay_lofo_matched_cells.py`, which writes
   `experiments/lofo_matched_cells_rank1.json`. Every number in 3.5, 3.6 and 3.7 — the
   matched-cell MedAEs, the paired-fold counts and Wilcoxon p-values, the bootstrap CIs,
   the per-column positivity counts, the suite table, and the allowlist arm — is read
   directly from that artifact.
8. Re-verify upstream parity against the pinned BenchPress commit:
   `scripts/verify_benchpress_parity.py`.

Relevant implementation anchors: completion `src/pathopress/completion.py:163`; greedy
selection `src/pathopress/probes.py:832`; pairwise ranking
`src/pathopress/ranking.py:134`; claim assertions `src/pathopress/maintenance.py:225`
and `:278` (which currently encode margin-5 invariants and should be revisited alongside
item 3 above).
