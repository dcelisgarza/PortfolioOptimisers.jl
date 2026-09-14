# High Order Prior

```@docs
HighOrderPriorEstimator
prior(pe::HighOrderPriorEstimator, X::MatNum,
               F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing;
               dims::Int = 1, kwargs...)
block_vec_pq
elimination_matrix
summation_matrix
dup_elim_sum_matrices
duplication_matrix
dup_elim_sum_view(args...)
dup_elim_sum_view(::MatNum, n)
PortfolioOptimisers.assemble_high_order_prior
PortfolioOptimisers.comoment_investable
PortfolioOptimisers.assert_matched_coverage
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
