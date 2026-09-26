```@meta
Description = "The prior family on the partial-fit seam, private API of PortfolioOptimisers.jl: PriorCarryState, sample_buffer, returns_buffer, prior_returns_buffer, …"
```

# The prior family on the partial-fit seam: private API

```@docs
PortfolioOptimisers.PriorCarryState
PortfolioOptimisers.sample_buffer(state::PortfolioOptimisers.PriorCarryState)
PortfolioOptimisers.returns_buffer
PortfolioOptimisers.prior_returns_buffer
PortfolioOptimisers.fold_carry
Base.copy(x::PortfolioOptimisers.PriorCarryState)
PortfolioOptimisers.needs_factor_returns
PortfolioOptimisers.combine_factor_answers
PortfolioOptimisers.assert_factor_returns
PortfolioOptimisers.fold_factor_argument
PortfolioOptimisers.fold_member
PortfolioOptimisers.read_member
PortfolioOptimisers.update_online_estimator(pe::Union{<:HighOrderPriorEstimator, <:BlackLittermanPrior})
```
