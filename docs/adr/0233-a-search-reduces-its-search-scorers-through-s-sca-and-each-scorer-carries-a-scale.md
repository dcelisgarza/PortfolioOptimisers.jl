---
status: accepted
---

# A search reduces its search scorers through `s_sca`, and each scorer carries a scale

## Context

Map #1980 lets `GridSearchCrossValidation` and `RandomisedSearchCrossValidation` order their
candidates with a vector of search scorers in `scorer`. ADR 0232 named the slot that reduces the
vector, `s_sca`, beside `r_sca`. Issue #1984 asked how the search combines the values of the
scorers for each candidate, what a function scorer in the vector does, and what `rank_scores`
holds when the reduction is `nothing`.

Five facts bore on the answer.

1. **A search scorer gives one value per candidate, bigger is better.** `candidate_scores` reads
   the `folds × candidates` score matrix and returns one value per column (ADR 0226). So every
   value is already oriented, and the rule of ADR 0231 for a vector of oriented values applies.
2. **The seam already reduces per candidate.** `scalarise` combines array values elementwise, so
   `scalarise(s -> candidate_scores(s, X), sca, scorers)` gives one reduced value per candidate. A
   `VectorToScalarMeasure` or a function reads the vector of `L` values of one candidate.
3. **A function scorer gave an index.** The released contract of `CrossValSearchScorer` was "a
   plain function that accepts a matrix and returns an integer". One index cannot be combined
   with other values, and it cannot order `n_best > 1` candidates. The constructor refused a
   function with `n_best > 1` for this reason alone.
4. **A risk measure carries a weight, and a search scorer carries none.** `oriented_member_score`
   multiplies the oriented value of each member by `r.settings.scale`. `HighestMeanScore` has no
   fields, so `SumScalariser` would add the values of the scorers with equal weights.
5. **Under `nothing`, each scorer orders its candidates apart.** The direction of the maintainer
   on map #1980 is that the search gives a ranking for each value per candidate. So two scorers
   keep different candidates, in different orders.

## Decision

**A search reduces the values of its search scorers per candidate through `s_sca`, after it
multiplies each value by the scale of its scorer. A function scorer gives what a scorer gives.
Under `s_sca = nothing`, `rank_scores` holds one entry for each scorer.**

| Item | Rule |
| --- | --- |
| The field | `s_sca::Option{<:Sca_VecToScaM}` on `GridSearchCrossValidation` and `RandomisedSearchCrossValidation`, with the default `SumScalariser()`, as `r_sca` (ADR 0232). No wrapper type holds the vector. |
| `scorer` | One search scorer, or an `AbstractVector` of them. A literal vector of a type and a function, such as `[HighestMeanScore(), f]`, has the element type `Any`, so the constructor checks each member and does not bind the element type. A wrong member is a fault of the configuration, so the constructor refuses it. |
| The orientation | Every value is bigger-is-better before the reduction (ADR 0226, ADR 0231). `MaxScalariser` and `MaxValue` give the best scorer of each candidate, and `MinScalariser` and `MinValue` the worst. `StdValue` gives the highest value to the largest spread. |
| The seam | A `Scalariser` reduces through `scalarise`, elementwise over the vectors of the scorers. A `VectorToScalarMeasure` or a function reduces the vector of `L` values of each candidate through `vec_to_real_measure`. |
| The scale | Each built-in search scorer holds `scale::Number = 1.0`, the same word and mechanism as `settings.scale` of a risk measure. The constructor refuses a non-finite scale, as `HierarchicalRiskMeasureSettings` does. The reduction multiplies the value of each member by its scale. A negative scale is a negative weight (ADR 0231). A user subtype with no `scale` field, and a function, read a scale of one through a fallback accessor. A caller who wants a scaled function writes the scale into the function. |
| A function scorer | It gives what a search scorer gives, in the same form: one value per column of the score matrix, bigger is better, or the index of one winner. So a raw function can do the job of any built-in scorer, for example `X -> vec(mean(X; dims = 1))` for `HighestMeanScore()`. The search reads the form from the type of the return of its one call. A vector is the new form, and an integer takes the path that a function took before. |
| An index-only member | A function that returns an index, and a subtype that defines only the index form, stay. Alone, each keeps one candidate, as before. In a vector under a reduction, the search refuses it when it runs, because an index has no value to combine. In a vector under `nothing`, it keeps its one index with `n_best = 1`, and its entry of `rank_scores` is `nothing`. |
| The `n_best` check | The refusal of a function with `n_best > 1` moves from the constructor to the run, because only the return of the function shows its form. It applies only to a return that is an index. The constructor keeps `n_best > 0`. |
| One scorer | When `scorer` holds one scorer, not a vector, `s_sca` and the scale are inert, as for one risk measure. No search of one scorer changes. |
| The shape with `nothing` | `rank_scores` is a vector of `L` entries in the order of `scorer`. Entry `l` is what scorer `l` alone gives: the values of the candidates that it keeps, in its own order, as `Option{<:VecNum}`. The shape follows the types of `scorer` and `s_sca`, never the count at run time (ADR 0227), so a vector of one scorer under `nothing` gives a vector of one entry. Each value keeps its scale, so a negative scale reverses the order of that scorer alone. |
| ADR 0120 | Unchanged. Every scorer reads the same finite columns, and a candidate with a non-finite entry enters no order. |
| The selection with `nothing` | Out of this ADR. The selection ticket of map #1980 decides what `opt`, `ranked` and `rank_idx` hold under `s_sca = nothing`, and the shape when `r_sca` and `s_sca` are both `nothing`. |

## Considered options

- **A wrapper type that holds the vector and its reduction.** It keeps `scorer` one value and
  `finite_candidate_order` unchanged. Rejected: ADR 0232 named the slot `s_sca` on the search,
  beside `r_sca`, so the two reductions of a search have one form.
- **Refuse a function in the vector by the bound `AbstractVector{<:CrossValidationSearchScorer}`.**
  Rejected by the maintainer: a function gives what a scorer gives, so it can give a value per
  candidate. The bound also refuses a literal vector of a type and a function, whose element type
  is `Any`.
- **A function that reads one column and returns one number.** It is the form of the `Function`
  member of `Sca_VecToScaM`, so `scorer = mean` would work. Rejected: a function must give what
  a scorer gives, and a scorer reads the whole matrix. A released index function also runs on a
  column, gives `1`, and ties every candidate with no error.
- **Remove the index-only form of a subtype, so that one contract holds.** Rejected by the
  maintainer: the change adds a capability and removes none, and the old path pays nothing for
  the new one.
- **No scale, with the weights in `s_sca = MeanValue(; w)` or in a function.** It changes no
  type. Rejected by the maintainer: a scorer takes the same word and mechanism as a risk measure.
- **A weight vector `s_w` on the search.** Rejected: a second place for a weight that a member
  of the vector already states, and a length check against `scorer`.
- **A `candidates × L` matrix under `nothing`.** Rejected: the rows need one order that the `L`
  scorers share, and `nothing` gives none. A matrix also cannot hold the `nothing` entry of an
  index-only member.

## Consequences

- **A breaking change, for the build.** The released docstring of `CrossValSearchScorer` says
  that a function returns an integer. The build widens it to the two forms. A released index
  function keeps its meaning. A released call of `HighestMeanScore()` keeps its meaning, because
  the scale defaults to `1.0` and is inert for one scorer. A user subtype keeps working with no
  `scale` field. The constructor no longer refuses a function with `n_best > 1`, so a released
  index function with `n_best > 1` now fails when the search runs. The migration guide names it.
- **A call that uses no new capability costs what it cost before.** For one scorer, the search
  makes the one call of the scorer that it made before, and the type of the return picks the
  path. The type of `scorer` and of `s_sca` picks the method of the reduction, so a search of one
  scorer does not reach it.
- **ADR 0226 is not on `main`, so its function rules are rewritten in place.** Its fallback now
  reads the return of a function, and its `n_best` refusal of a function moves to the run.
