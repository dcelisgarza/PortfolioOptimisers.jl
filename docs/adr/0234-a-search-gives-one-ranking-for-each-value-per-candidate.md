---
status: accepted
---

# A search gives one ranking for each value per candidate

## Context

Map #1980 lets a search score with a vector of risk measures and a vector of search scorers. ADR
0232 keeps one score matrix per measure under the per-value form of `r_sca`, and ADR 0233 keeps one
entry of `rank_scores` per scorer under the per-value form of `s_sca`. The per-value form is a
singleton type that #2001 names. #1985 spelled it `nothing`, and #2069 (ADR 0238) made `nothing`
give the sum on every scalariser slot. Both ADRs left the selection open: what `opt`,
`idx`, `ranked` and `rank_idx` hold when a candidate has more than one value. The direction of the
maintainer on the map is that the search does not refuse this case. It gives a ranking and an
`opt` for each value per candidate. Issue #1985 asked for the shape of the Result, the shape in
each mixed case, the order of the reductions, the rule of `n_best`, and whether the search takes a
named refit as sklearn does.

Four facts bore on the answer.

1. **The earlier ADRs keep the per-value data in one Result.** `test_scores` holds `K` matrices
   (ADR 0232), and `rank_scores` holds `L` entries (ADR 0233).
2. **The two orders of a reduction give different numbers.** For one candidate, `r_sca` combines
   the `K` measures in each fold, and then the search scorer reduces the folds. The scorer can
   also reduce the folds of each measure first, and then a scalariser combines the `K` values.
   For `SumScalariser` the two orders agree. For `MaxScalariser`, `MinScalariser`,
   `LogSumExpScalariser` or a function they do not: the mean over the folds of the best measure is
   not the best measure of the means.
3. **The two stages are exclusive.** When `r_sca` combines in each fold, one matrix is left, and a
   combination after the folds has nothing to combine.
4. **A search keeps configurations, not fits (ADR 0226).** The refit of sklearn fits the winner on
   the whole window. Here the caller does that with `optimise`.

## Decision

**A search gives one ranking for each value per candidate, in one Result. `r_sca =
PerCandidate(sca)` combines the measures once per candidate, after the search scorer reduces the
folds.**

### The stage of `r_sca`

| `r_sca` | What it combines | `test_scores` |
| --- | --- | --- |
| `sca`, a `Sca_VecToScaM`, or `nothing`, which gives the sum | The `K` measures of one row: one fold, or one path in the combinatorial form. Then the scorer reduces the rows. ADR 0232. | One matrix. |
| `PerCandidate(sca)` | The `K` values of one candidate. Each value is what the scorer gives for one matrix over all its rows. | `K` matrices. |
| The per-value singleton (#2001) | Nothing. Each measure keeps its own ranking. | `K` matrices. |

For one candidate, two measures, three folds, `MaxScalariser` and `HighestMeanScore`:

```text
            fold 1  fold 2  fold 3   mean
measure 1      3       0       0      1
measure 2      0       2       1      1

r_sca = MaxScalariser():               max per fold [3, 2, 1], then mean = 2
r_sca = PerCandidate(MaxScalariser()): mean per measure [1, 1], then max = 1
```

The first asks how good the best measure is in each fold. The second asks how good the best
measure is over all the folds.

| Item | Rule |
| --- | --- |
| `PerCandidate` | A wrapper that holds one `Sca_VecToScaM`. Exported. The bound of `r_sca` widens to take it. For one measure it is inert, as `r_sca` is. |
| The order | The scorer reduces the folds or the paths of each matrix, giving `v[k, l]` for measure `k` and scorer `l`. `PerCandidate` combines `v[:, l]` for each scorer. Then `s_sca` combines the `L` results. So the measures combine before the scorers in both stages, and `s_sca` keeps the definition of ADR 0233. |
| The scale | Each value of a matrix holds the scale of its measure (ADR 0231), so `PerCandidate` reads the scale as the weight of the measure. |
| An index-only scorer | It gives no value to combine, so the search refuses it under `PerCandidate` when it runs, as ADR 0233 does under `s_sca`. |
| Other slots | `NearestQuantilePrediction` and `FoldRiskScorer` have no scorer stage, so their `sca` keeps the bound `Option{<:Sca_VecToScaM}` and refuses `PerCandidate`. |

### The shape of the Result

One `SearchCrossValidationResult`. `opt`, `idx`, `ranked`, `rank_idx` and `rank_scores` hold one
entry per value. The shape follows the types of `r`, `r_sca`, `scorer` and `s_sca`, never the count
at run time (ADR 0227).

| `r_sca` | `s_sca` | Values per candidate | Container of each field |
| --- | --- | --- | --- |
| `sca`, `nothing`, `PerCandidate`, or one measure | `sca`, `nothing`, or one scorer | `1` | Today's fields. |
| `sca`, `nothing`, `PerCandidate`, or one measure | per-value | `L` | A vector, in the order of `scorer`. |
| per-value | `sca`, `nothing`, or one scorer | `K` | A vector, in the order of `r`. |
| per-value | per-value | `K × L` | A `K × L` matrix. Row `k` is measure `k`, column `l` is scorer `l`. |

So `res.opt[k, l]` is the winner of measure `k` under scorer `l`, and `res.opt[:, l]` holds the
winners of every measure under scorer `l`. The constructor checks `opt[v] == ranked[v][1]` and
`idx[v] == rank_idx[v][1]` for each entry that is not empty.

### `n_best` and the failed candidates

| Item | Rule |
| --- | --- |
| `n_best` | Each ranking keeps `min(n_best, finite count)` candidates, so the rankings can differ in length. An index-only scorer keeps one candidate (ADR 0233). |
| ADR 0120 under the per-value form | Holds for each matrix apart (ADR 0232). A candidate can enter the ranking of one measure and stay out of the ranking of another. |
| ADR 0120 under `PerCandidate` | A candidate enters only when it is finite in every matrix, because a value of fewer than `K` measures does not compare with a value of `K`. |
| An empty entry | Under the per-value form of `r_sca`, a matrix with no finite column gives an empty entry: `opt[v]` and `idx[v]` are `nothing`, `ranked[v]` and `rank_idx[v]` are empty, and `rank_scores[v]` is an empty vector, or `nothing` for an index-only scorer. Only the per-value form of `r_sca` can give one empty entry beside entries that are not empty. |
| Every entry empty | The search throws the `IsNonFiniteError` of `finite_candidate_order`, as for one value. The constructor checks that at least one entry holds a candidate. No search of one value changes. |

### No refit

The search takes no field that names one value to select by. A caller who wants the winner of value
`v` fitted on the whole window calls `optimise(res.opt[v], rd)`, as for one value. The docstring
of the search states this as the form of the `refit` of sklearn.

## Considered options

- **A vector of Results, one per value.** Each Result keeps today's field types. Rejected: ADR 0232
  and ADR 0233 already keep the per-value data in one Result, and each Result would repeat the
  shared `test_scores`, `lens_grid` and `val_grid`.
- **A third field for the combination after the folds.** Rejected: it acts only when `r_sca` holds
  the per-value form, so a configuration could set a field that has no effect, and the constructor would
  need a conflict check.
- **No combination after the folds.** Rejected: it drops the reason that the map gives for
  the per-value form on `r`.
- **The scorers before the measures after the folds.** Rejected: the measures combine before the
  scorers in the stage of each fold, and the order of the two stages then differs.
- **The names `PerPath` and `AfterFolds`.** Rejected. `PerPath` names the stage of each row in the
  combinatorial form, which is the other stage. `AfterFolds` is wrong in the combinatorial form,
  where the rows are paths.
- **Nested vectors `opt[k][l]` for `K × L` values.** The form of `train_scores[k][p]` (ADR 0232).
  Rejected: that form is nested because its inner vector of paths was already released, and a
  slice across the measures needs a comprehension.
- **A throw when one matrix has no finite column.** It keeps `opt[v] == ranked[v][1]` for every
  entry. Rejected by the maintainer: the other measures still have a winner, and the search keeps
  it.
- **No throw when every entry is empty.** Rejected: a Result with no candidate gives no data, and
  the search of one value throws in that case.
- **A named refit, as sklearn's `refit = "metric"`.** Rejected by the maintainer: the refit is a
  call to `optimise` outside the search.

## Consequences

- **The build tickets of map #1980.** The code holds none of this decision until they land.
  Ticket #2001 builds the per-value singleton, the shape of the Result under it on `r_sca` or
  `s_sca`, the empty entry and its checks. Ticket #2002 builds `PerCandidate`. Ticket #2003 shows a search
  with a vector of risk measures and a vector of search scorers in the tuning example and the
  user guide.
- **A breaking change, for the build.** Under the per-value form of `r_sca` or `s_sca` with a vector,
  `opt`, `idx`, `ranked`, `rank_idx` and `rank_scores` change type. These are new configurations,
  so they reach no released caller. The bound of `r_sca` widens to take `PerCandidate`.
- **A call that uses no new capability costs what it cost before.** The types of the four fields
  pick the method of the selection, so a search of one value takes the path of today.
- **No reader in the library changes.** `Resume` refuses a search Result, and no plot recipe reads
  one.
