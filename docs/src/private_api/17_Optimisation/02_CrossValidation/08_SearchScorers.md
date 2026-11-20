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
assert_reduced_values
candidate_count
```

Under [`PerCandidate`](@ref), the search scorer reduces the matrix of each risk measure, and the rule combines the values of each candidate.

```@docs
measure_candidate_values
```

Under [`EachValue`](@ref), the search gives one ranking for each value per candidate.

```@docs
finite_score_columns
scorer_ranking
empty_ranking
measure_rankings
ranked_candidates
allocate_search_predictions
ranked_predictions
store_search_prediction!
assert_rank_predictions
```
