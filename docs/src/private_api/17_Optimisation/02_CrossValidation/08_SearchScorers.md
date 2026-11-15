```@meta
Description = "Search scorers, private API of PortfolioOptimisers.jl: CrossValSearchScorer, finite_candidate_order, assert_candidate_scores, candidate_values, …"
```

# Search scorers: private API

```@docs
CrossValSearchScorer
finite_candidate_order
assert_candidate_scores
candidate_values
member_candidate_values
reduce_scorer_values
```

Under [`EachValue`](@ref), the search gives one ranking for each value per candidate.

```@docs
finite_score_columns
scorer_ranking
empty_ranking
measure_rankings
ranked_candidates
```
