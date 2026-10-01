```@meta
Description = "Prior partial fit, private API of PortfolioOptimisers.jl: PriorCarryState, sample_buffer, returns_buffer, prior_returns_buffer, fold_carry, Base.copy, …"
```

# Prior partial fit: private API

```@docs
PortfolioOptimisers.PriorCarryState
PortfolioOptimisers.sample_buffer(state::PortfolioOptimisers.PriorCarryState)
PortfolioOptimisers.returns_buffer
PortfolioOptimisers.prior_returns_buffer
PortfolioOptimisers.fold_carry
Base.copy(x::PortfolioOptimisers.PriorCarryState)
PortfolioOptimisers.buffer_prior
PortfolioOptimisers.needs_factor_returns
PortfolioOptimisers.combine_factor_answers
PortfolioOptimisers.reads_panel_fields
PortfolioOptimisers.assert_factor_returns
PortfolioOptimisers.assert_prior_fold_returns
PortfolioOptimisers.refit_prior_step
PortfolioOptimisers.refit_prior_fold
PortfolioOptimisers.refit_step_kwargs
PortfolioOptimisers.step_panel_fields
PortfolioOptimisers.fold_factor_argument
PortfolioOptimisers.fold_member
PortfolioOptimisers.read_member
PortfolioOptimisers.assert_carry_step_panel
PortfolioOptimisers.step_active_kwargs
PortfolioOptimisers.update_online_estimator(pe::Union{<:HighOrderPriorEstimator, <:BlackLittermanPrior})
```
