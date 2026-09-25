```@meta
Description = "The optimiser on the partial-fit seam, private API of PortfolioOptimisers.jl: returns_buffer, prior_returns_buffer, step_active_mask, fold_prior, …"
```

# The optimiser on the partial-fit seam: private API

```@docs
PortfolioOptimisers.returns_buffer
PortfolioOptimisers.prior_returns_buffer
PortfolioOptimisers.step_active_mask
PortfolioOptimisers.fold_prior
PortfolioOptimisers.fold_context
PortfolioOptimisers.fold_returns
PortfolioOptimisers.online_state_seed(::Union{<:EqualWeighted, <:RandomWeighted, <:BestConstantRebalancedPortfolio}, max_history::PortfolioOptimisers.Option{<:Integer})
PortfolioOptimisers.update_online_member
PortfolioOptimisers.update_online_estimator(opt::PortfolioOptimisers.JuMPOptimisationEstimator)
PortfolioOptimisers.online_unreached_path
PortfolioOptimisers.assert_stateless_schedule
PortfolioOptimisers.assert_stateless_prior
PortfolioOptimisers.assert_online_entry(::TimeDependent)
PortfolioOptimisers.assert_online_fee_source(::Any, ::Any)
PortfolioOptimisers.returns_result(host::Union{<:JuMPOptimiser, <:HierarchicalOptimiser, <:InverseVolatility, <:NestedClustered, <:Stacking, <:SubsetResampling})
PortfolioOptimisers.held_timestamps
PortfolioOptimisers.readout_without_state
PortfolioOptimisers.online_readout
PortfolioOptimisers.show_fields(opt::Union{<:JuMPOptimiser, <:HierarchicalOptimiser, <:InverseVolatility, <:EqualWeighted, <:RandomWeighted, <:NestedClustered, <:Stacking, <:SubsetResampling})
```
