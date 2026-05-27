# High Order Prior

```@docs
HighOrderPriorEstimator
Base.getproperty(obj::HighOrderPriorEstimator, sym::Symbol)
factory(pe::HighOrderPriorEstimator, w::ObsWeights)
prior(pe::HighOrderPriorEstimator, X::MatNum,
               F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
prior_view(pr::HighOrderPriorEstimator, rd)
block_vec_pq
elimination_matrix
summation_matrix
dup_elim_sum_matrices
duplication_matrix
dup_elim_sum_view(args...)
dup_elim_sum_view(::MatNum, n)
```
