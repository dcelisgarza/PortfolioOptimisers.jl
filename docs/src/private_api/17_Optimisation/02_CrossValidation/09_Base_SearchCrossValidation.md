```@meta
Description = "Base search cross validation, private API of PortfolioOptimisers.jl: AbstractSearchCrossValidationEstimator, assert_search_scorer, assert_search_reduction, …"
```

# Base search cross validation: private API

```@docs
AbstractSearchCrossValidationEstimator
assert_search_scorer
assert_search_reduction
search_result
assert_search_entry
search_candidate
assert_search_candidates(::Any, ::Any, ::Any)
score_rows
score_path_ids
fold_train_returns
candidate_train_score
write_candidate_scores!
allocate_search_scores
search_score_array
parse_lens
assert_search_grid_cap
_expr_to_lens
_eval_index
expr_to_lens_chain
```
