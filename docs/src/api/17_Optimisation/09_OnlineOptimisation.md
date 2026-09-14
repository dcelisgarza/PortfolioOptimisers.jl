# The optimiser on the partial-fit seam

An optimiser takes the online step in two verbs and one forward. `partial_fit!(opt, rd)` folds the observations of a carrier into the **prior alone** and records the rest of the carrier in a [`PortfolioOptimisers.ReturnsBufferState`](@ref); `optimise(opt)` with no returns rebuilds the carrier from the state, swaps the folded prior for its read-out and runs the ordinary batch path, so everything above the prior — the clustering estimator, the constraint estimators, every uncertainty set, a meta-optimiser's inner optimisers — is fitted exactly as batch fits it. The read-out is pure, so the fallback chain walks unchanged. ADR 0137 records the decision.

```@docs
PortfolioOptimisers.returns_buffer
PortfolioOptimisers.prior_returns_buffer
PortfolioOptimisers.step_active_mask
PortfolioOptimisers.fold_prior
PortfolioOptimisers.fold_context
PortfolioOptimisers.fold_returns
PortfolioOptimisers.partial_fit!(opt::PortfolioOptimisers.JuMPOptimisationEstimator, rd::ReturnsResult)
PortfolioOptimisers.online_state_seed(::Union{<:EqualWeighted, <:RandomWeighted}, max_history::PortfolioOptimisers.Option{<:Integer})
PortfolioOptimisers.update_online_member
PortfolioOptimisers.update_online_estimator(opt::PortfolioOptimisers.JuMPOptimisationEstimator)
PortfolioOptimisers.assert_stateless_schedule
PortfolioOptimisers.assert_stateless_prior
PortfolioOptimisers.assert_online_entry
PortfolioOptimisers.returns_result(host::Union{<:JuMPOptimiser, <:HierarchicalOptimiser, <:InverseVolatility, <:NestedClustered, <:Stacking, <:SubsetResampling})
PortfolioOptimisers.held_timestamps
PortfolioOptimisers.readout_without_state
PortfolioOptimisers.online_readout
optimise(opt::PortfolioOptimisers.OptimisationEstimator; kwargs...)
PortfolioOptimisers.show_fields(opt::Union{<:JuMPOptimiser, <:HierarchicalOptimiser, <:InverseVolatility, <:EqualWeighted, <:RandomWeighted, <:NestedClustered, <:Stacking, <:SubsetResampling})
```
