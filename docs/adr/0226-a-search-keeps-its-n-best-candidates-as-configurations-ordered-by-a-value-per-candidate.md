---
status: accepted
---

# A search keeps its `n_best` best candidates as configurations, ordered by a value per candidate

## Context

`search_cross_validation` scores every candidate of its grid, but its Result named one candidate:
`opt` and `idx`. A caller who wanted the best `n` candidates had to sort the columns of
`test_scores` by hand and build each candidate again from its lenses. Issue
[#1937](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1937) found that the second
step was not possible from the Result alone. The Result kept `opt`, not the estimator that the
search was given, and `search_candidate(res.opt, res.lens_grid[k], res.val_grid[k])` gives
candidate `k` only when every candidate sets the same lenses. In a grid that joins parameter sets
with different keys, candidate `k` of the second set kept the value of the first key from the
winner.

The scorer gave one index. An index can pick one candidate, but it cannot order `n` of them.

## Decision

### 1. The scorer gives a value per candidate: `candidate_scores`

`candidate_scores(scorer, X)` is a public verb of the `# Interfaces` of
`CrossValidationSearchScorer`. It returns one value per column of the score matrix, bigger is
better. `HighestMeanScore` returns the column means. Its index form `HighestMeanScore()(X)` does
not change, and it equals `argmax` of its values.

The fallback, on the alias `CrossValSearchScorer`, returns `nothing`: the scorer gives no value per
candidate. A subtype that defines only the index form takes the fallback. The search then calls
the index form and keeps one candidate, as it did before, and it refuses `n_best > 1` with an
`ArgumentError` that names the type and the method to add. A function scorer gives what a subtype
gives, in the same form: one value per column, or the index of one winner. The search reads the
form from the type of the return of its one call, and an index takes the same path as the index
form of a subtype (ADR 0233).

Issue #1937 asked for a fallback that throws. A fallback that returns `nothing` keeps the same
refusal at `n_best > 1`, and it lets the search read the values of every scorer that has them,
also at `n_best = 1`. So `rank_scores` holds the mean of the winner under the default scorer, and
rank 1 comes from the same order at every `n_best`. The library spells "no value" with `nothing`
(ADR 0222).

### 2. `n_best` on both search estimators

`GridSearchCrossValidation` and `RandomisedSearchCrossValidation` take `n_best::Integer = 1` after
`scorer`. The constructor checks `n_best > 0`. A function scorer that returns an index is refused
with `n_best > 1` when the search runs, because only its return shows its form (ADR 0233). The
randomised search forwards `n_best` to
the grid search of its trials. Its trials come from `rng`, so its order is the same in two runs only
when `seed` is set.

### 3. The order, the cap and the failed candidates

`finite_candidate_order(scorer, test_scores, n_best)` replaces `finite_candidate_index`, so the
rule of ADR 0120 is written once. Only a column whose every entry is finite enters the order. The
order is `cols[sortperm(candidate_scores(scorer, view(test_scores, :, cols)); rev = true)]`, cut at
`min(n_best, length(cols))`. The sort is stable, so a tie keeps grid order, as `argmax` does. The
cap gives no warning: `test_scores` shows each failed column.

### 4. The Result keeps configurations, not fits

`SearchCrossValidationResult` has four more fields:

| Field | Content |
| --- | --- |
| `ranked` | The kept candidates, best first, each built from the estimator or the `Pipeline` that the search was given, through the lenses of its own grid point. |
| `rank_idx` | The column of `test_scores` of each kept candidate. |
| `rank_scores` | `candidate_scores` of each kept candidate, or `nothing`. |
| `path_ids` | The path of each row of `test_scores`: `cv.path_ids` under `MultipleRandomised`, `1:M` under `CombinatorialCrossValidation`, `nothing` for a scheme with one path. |

`opt` is `ranked[1]` and `idx` is `rank_idx[1]`, and the constructor refuses fields that do not
agree. At `n_best = 1` the Result holds the `opt`, `idx`, `test_scores` and `train_scores` of the
released search. The Result keeps no summary of the spread of a candidate. A caller computes it
from `test_scores[:, rank_idx]`.

The four search methods, on an optimiser and on a `Pipeline`, batch and combinatorial, build the
Result through one internal function, `search_result`. The score loops do not change.

## Rejected options

- **A scorer that returns a full permutation.** It can rank by a rule with no single number, but
  the Result then has no value to show for a candidate.
- **Always rank by the mean, whatever the scorer.** Rank 1 could then differ from `opt` under a
  custom scorer.
- **Keep the fitted fold results, or a refit on the full window, of each kept candidate.** The fold
  fits cost `n_best × M` results of memory, and a caller who allocates needs a fit on the full
  window, not the fold fits. The search does not refit, and this decision adds no refit.
- **A separate function over the Result.** It would need the base estimator as a second argument,
  because the Result does not keep it, and the ranking would live apart from the scorer that picks
  `idx`.
- **Score each path of a `MultipleRandomised` scheme once, as the combinatorial arm does.** It
  changes the released `test_scores` and the released pick. `path_ids` gives the view per path
  with no change to the scores.

## Consequences

- The positional constructors of `GridSearchCrossValidation`, `RandomisedSearchCrossValidation`
  and `SearchCrossValidationResult` take more arguments. The keyword constructors keep every
  earlier keyword, and the new keywords of the Result have defaults.
- A scorer that ranks by spread, for example the mean less a multiple of the standard deviation,
  is one more subtype that defines `candidate_scores`.
