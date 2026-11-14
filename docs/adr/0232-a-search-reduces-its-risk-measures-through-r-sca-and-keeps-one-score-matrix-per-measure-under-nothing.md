---
status: accepted
---

# A search reduces its risk measures through `r_sca`, and keeps one score matrix per measure under the per-value form

## Context

Map #1980 lets `GridSearchCrossValidation` and `RandomisedSearchCrossValidation` score with a
vector of risk measures in `r`. ADR 0231 fixed the score: each measure is oriented, multiplied by
its scale, and then reduced by a `Sca_VecToScaM`, or kept apart under the per-value form. Issue
#1983 asked where a search holds that reduction, what its default is, and what shape the scores
take under the per-value form. #1983 spelled that form `nothing`. #2069 (ADR 0238) made `nothing`
give the sum on every scalariser slot, so the form takes its own singleton type, which #2001
names. This ADR calls it "the per-value form".

Four facts bore on the answer.

1. **The search forwards `kwargs` to the reader of a risk.** Each fold wrote
   `sgn * expected_risk(r, p; kwargs...)`, so a reduction in `kwargs.sca` was the first option.
   `NearestQuantilePrediction` read its reduction from `r_kwargs.sca` in the same way (#1982).
2. **A search holds two reductions.** #1984 adds a reduction for a vector of search scorers. A
   field named `sca` alone does not say which vector it reduces.
3. **The scorer and the order read one matrix.** `finite_candidate_order` and every
   `CrossValidationSearchScorer` read a `folds × candidates` matrix, and ADR 0120 keeps a
   candidate with a non-finite entry out of the order.
4. **The six methods did not share their writer.** The two batch forms wrote through
   `write_candidate_scores!`, but the two combinatorial forms had their own loop, and each of the
   four grid methods allocated its own matrix. The two randomised forms copy their fields into a
   `GridSearchCrossValidation`.

## Decision

**A search holds the reduction of its risk measures in a field `r_sca`, with the default
`nothing`, which gives the sum (ADR 0238). Under the per-value form, the search keeps one score
matrix for each measure.**

| Item | Rule |
| --- | --- |
| The field | `r_sca::Option{<:Sca_VecToScaM}` on `GridSearchCrossValidation` and `RandomisedSearchCrossValidation`. The bound also takes the per-value singleton of #2001 and `PerCandidate(sca)`, which combines the measures once per candidate after the search scorer (ADR 0234). The constructor refuses a `kwargs.sca`, and the error names `r_sca`. The reduction of the vector of search scorers is `s_sca` (ADR 0233). |
| The name of a slot | A type that holds one reduction names its field `sca`. A type that holds two names them `r_sca`, for the risk measures, and `s_sca`, for the search scorers. So `NearestQuantilePrediction` and `FoldRiskScorer` each hold a field `sca`, and `NearestQuantilePrediction` refuses `r_kwargs.sca`. |
| The default | `nothing`, which gives the sum of `SumScalariser`, as `expected_risk`, `oriented_score` and every reader bound to `BaseRM_VecBaseRM` (ADR 0238). For one measure the slot is inert, so no search of one measure changes. |
| The orientation | ADR 0231. Each measure is oriented before the reduction, so a higher score is better. The local `sgn` of each method goes away. A train score is oriented in the same way, so the build gives `oriented_score` a form for a fitted result. |
| The shape under the per-value form | `test_scores` is a vector of `K` matrices, each `folds × candidates`, in the order of `r`. `train_scores` takes the same shape, and in the combinatorial form `train_scores[k][p]` is path `p` of measure `k`. The shape follows the types of `r` and `r_sca`, never the count at run time (ADR 0227), so a vector of one measure under the per-value form gives a vector of one matrix. |
| `path_ids` | One vector. Every matrix has the same rows. |
| ADR 0120 | Holds for each matrix apart. Each matrix goes through `finite_candidate_order` unchanged. |
| The selection under the per-value form | ADR 0234. `opt`, `idx`, `ranked`, `rank_idx` and `rank_scores` hold one entry per measure. |
| The six methods | One writer and one allocator. The combinatorial score becomes a method of `write_candidate_scores!`, and one helper allocates a matrix or a vector of `K` matrices. The two randomised forms forward `r_sca`. |
| The keyword form of the score | `oriented_score(r, p; sca, kwargs...)` is removed. Every caller holds its reduction and calls `oriented_score(sca, r, p; kwargs...)`. `sort_by_measure(sca, ppred, r; kwargs...)` and `quantile_by_measure(sca, ppred, r, q; r_kwargs, q_kwargs)` take the reduction as their first argument in the same way, and they have no keyword form. #1997 measured the keyword form against the positional form: the time, the allocations and the count of runtime dispatches were equal in each case. So the form follows the rule for keyword wrappers in `src/CLAUDE.md`. `quantile_by_measure` and `NearestQuantilePrediction` refuse a `sca` in `r_kwargs`, and `sort_by_measure` refuses a `sca` in its `kwargs`, because a reader of a risk would ignore it for one measure. |
| `nothing` with a vector | `NearestQuantilePrediction`, `sort_by_measure`, `quantile_by_measure` and `FoldRiskScorer` admit `sca = nothing` with a vector, and it gives the sum (ADR 0238). They take no per-value singleton, because a sort, a quantile and a selector need one score for each path or asset. |

## Considered options

- **The reduction in `kwargs.sca`.** It was the precedent of `NearestQuantilePrediction`, and it
  needs no conflict check. Rejected by the maintainer: a field is visible in the field list and in
  `show`, and the prefix tells the two reductions of a search apart.
- **The per-value form as the default.** It is the multi-metric form of sklearn. Rejected: every
  reader of a vector of measures in the library sums by default, and a search that only changes
  `r` from one measure to a vector would change the type of its Result.
- **`nothing` as the spelling of the per-value form.** It was the first decision of #1983.
  Replaced by ADR 0238: `nothing` gives the sum on every scalariser slot.
- **A `folds × candidates × K` array.** It is compact, but ADR 0230 puts the observations of a
  rank-3 array on its last axis, and no slice of a rank-3 array with the folds last is the
  `folds × candidates` matrix that a scorer reads.
- **A collection keyed by the measure.** A Performance Summary keys its extra measures by a name
  that the caller gives (ADR 0227). The `r` of a search has no names, and two measures of one type
  with different parameters have no distinct key.
- **Edit the four grid methods one by one.** Rejected: four copies of one rule.

## Consequences

- **The build tickets of map #1980.** The code holds none of this decision until they land.
  Ticket #1997 gives `NearestQuantilePrediction` its field `sca`, gives `sort_by_measure` and
  `quantile_by_measure` their keyword `sca`, and removes the keyword form of `oriented_score`.
  Ticket #1998 gives the two searches `r_sca`, the one writer and the one allocator.
  Ticket #1999 gives `FoldRiskScorer` its field `sca`, with the default `nothing` (ADR 0238). A
  vector gives each asset its oriented score on each window, so `bigger_is_better` of the scorer
  reads `true` as the orientation of the values. One measure keeps its raw value and its own
  orientation, so no number of one measure moves. Ticket #2001 builds the per-value
  singleton, its shape and its selection (ADR 0234).
- **A breaking change, for the build.** The released `quantile_by_measure` and
  `NearestQuantilePrediction` pass `r_kwargs` to `expected_risk`, so a released caller can write
  `r_kwargs = (sca = MaxScalariser(),)`. That spelling is now refused, and the error names the
  argument `sca` of `quantile_by_measure` and the field `sca` of `NearestQuantilePrediction`. The
  build names the break in the migration guide, beside the `sign` that ADR 0231 removed. A
  released call `sort_by_measure(ppred, r; …)` or `quantile_by_measure(ppred, r, q; …)` raises a
  `MethodError`, because the reduction is now the first argument. The Result of a search holds a vector of matrices in `test_scores` and
  `train_scores` under the per-value form, a new configuration that reaches no released caller.
- **No search of one measure changes.** The slot is inert for one measure, and the orientation of
  one measure is the `sgn` that the search applied before.
- **A call that uses no new capability costs what it cost before.** The type of `r_sca` decides
  the shape, so the compiler picks the method of the allocator and of the writer, and a search of
  one measure keeps one matrix. #1998 measured the old and the new bodies of the writer, the
  allocator and the vector score in one slotted process, for `ConditionalValueatRisk` and
  `MeanReturn`, with and without train scores, batch and combinatorial, at 160 × 8 and
  2000 × 50. The allocations were equal or lower in every case. The times were equal within
  noise: the three cases whose first sample was slower gave the same minimum over five
  alternated rounds.

This ADR changes the form of the score in ADR 0231: the keyword form of `oriented_score` is
removed.
