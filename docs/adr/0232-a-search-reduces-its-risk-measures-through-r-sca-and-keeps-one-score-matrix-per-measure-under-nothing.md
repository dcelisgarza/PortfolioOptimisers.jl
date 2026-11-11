---
status: accepted
---

# A search reduces its risk measures through `r_sca`, and keeps one score matrix per measure under `nothing`

## Context

Map #1980 lets `GridSearchCrossValidation` and `RandomisedSearchCrossValidation` score with a
vector of risk measures in `r`. ADR 0231 fixed the score: each measure is oriented, multiplied by
its scale, and then reduced by a `Sca_VecToScaM`, or kept apart under `nothing`. Issue #1983 asked
where a search holds that reduction, what its default is, and what shape the scores take when the
reduction is `nothing`.

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
`SumScalariser()`. Under `r_sca = nothing`, the search keeps one score matrix for each measure.**

| Item | Rule |
| --- | --- |
| The field | `r_sca::Option{<:Sca_VecToScaM}` on `GridSearchCrossValidation` and `RandomisedSearchCrossValidation`. The bound also takes `PerCandidate(sca)`, which combines the measures once per candidate after the search scorer (ADR 0234). The constructor refuses a `kwargs.sca`, and the error names `r_sca`. The reduction of the vector of search scorers is `s_sca` (ADR 0233). |
| The name of a slot | A type that holds one reduction names its field `sca`. A type that holds two names them `r_sca`, for the risk measures, and `s_sca`, for the search scorers. So `NearestQuantilePrediction` and `FoldRiskScorer` each hold a field `sca`, and `NearestQuantilePrediction` refuses `r_kwargs.sca`. |
| The default | `SumScalariser()`, as `expected_risk`, `oriented_score` and every reader bound to `BaseRM_VecBaseRM`. For one measure the slot is inert, so no search of one measure changes. |
| The orientation | ADR 0231. Each measure is oriented before the reduction, so a higher score is better. The local `sgn` of each method goes away. A train score is oriented in the same way, so the build gives `oriented_score` a form for a fitted result. |
| The shape with `nothing` | `test_scores` is a vector of `K` matrices, each `folds × candidates`, in the order of `r`. `train_scores` takes the same shape, and in the combinatorial form `train_scores[k][p]` is path `p` of measure `k`. The shape follows the types of `r` and `r_sca`, never the count at run time (ADR 0227), so a vector of one measure under `nothing` gives a vector of one matrix. |
| `path_ids` | One vector. Every matrix has the same rows. |
| ADR 0120 | Holds for each matrix apart. Each matrix goes through `finite_candidate_order` unchanged. |
| The selection with `nothing` | ADR 0234. `opt`, `idx`, `ranked`, `rank_idx` and `rank_scores` hold one entry per measure. |
| The six methods | One writer and one allocator. The combinatorial score becomes a method of `write_candidate_scores!`, and one helper allocates a matrix or a vector of `K` matrices. The two randomised forms forward `r_sca`. |
| The keyword form of the score | `oriented_score(r, p; sca, kwargs...)` is removed. Every caller holds its reduction and calls `oriented_score(sca, r, p; kwargs...)`. `sort_by_measure` and `quantile_by_measure` take a keyword `sca = SumScalariser()`, as `expected_risk` does. `quantile_by_measure` and `NearestQuantilePrediction` refuse a `sca` in `r_kwargs`, because a reader of a risk would ignore it for one measure. |

## Considered options

- **The reduction in `kwargs.sca`.** It was the precedent of `NearestQuantilePrediction`, and it
  needs no conflict check. Rejected by the maintainer: a field is visible in the field list and in
  `show`, and the prefix tells the two reductions of a search apart.
- **`nothing` as the default.** It is the multi-metric form of sklearn. Rejected: every reader of a
  vector of measures in the library sums by default, and a search that only changes `r` from one
  measure to a vector would change the type of its Result.
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
  Ticket #1999 gives `FoldRiskScorer` its field `sca`. Ticket #2001 builds the shape under
  `nothing` and its selection (ADR 0234).
- **A breaking change, for the build.** The released `quantile_by_measure` and
  `NearestQuantilePrediction` pass `r_kwargs` to `expected_risk`, so a released caller can write
  `r_kwargs = (sca = MaxScalariser(),)`. That spelling is now refused, and the error names the
  keyword `sca` of `quantile_by_measure` and the field `sca` of `NearestQuantilePrediction`. The
  build names the break in the migration guide, beside the `sign` that ADR 0231 removed. A
  released call `sort_by_measure(ppred, r; sca = …)` keeps its meaning, because the new keyword
  has the same spelling. The Result of a search holds a vector of matrices in `test_scores` and
  `train_scores` under `r_sca = nothing`, a new configuration that reaches no released caller.
- **No search of one measure changes.** The slot is inert for one measure, and the orientation of
  one measure is the `sgn` that the search applied before.
- **A call that uses no new capability costs what it cost before.** The type of `r_sca` decides
  the shape, so the compiler picks the method of the allocator and of the writer, and a search of
  one measure keeps one matrix.

This ADR changes the form of the score in ADR 0231: the keyword form of `oriented_score` is
removed.
