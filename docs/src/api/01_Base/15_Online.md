# The online step

## The online step

An estimator with no exact incremental fold keeps the observations it has seen in a [`PortfolioOptimisers.SampleBufferState`](@ref), and [`Online`](@ref) is the configuration that seeds one. The wrapper is transient: [`PortfolioOptimisers.update_online_estimator`](@ref) resolves it at warm-up, so no wrapper survives into the run. The buffer carries the per-observation masks of a [`CoveragePolicy`](@ref) and the factor observations of a factor prior beside its rows, each fixed by the first append, so one buffer serves every estimator that refits.

```@docs
PortfolioOptimisers.SampleBufferState
PortfolioOptimisers.assert_sample_buffer_state
PortfolioOptimisers.assert_buffer_mask_shape
PortfolioOptimisers.assert_buffer_factor_shape
PortfolioOptimisers.buffer_rows_view
PortfolioOptimisers.sample_buffer
PortfolioOptimisers.sample_buffer_kwargs
PortfolioOptimisers.factor_buffer
PortfolioOptimisers.assert_sample_buffer(est::Union{<:PortfolioOptimisers.AbstractEstimator, <:StatsBase.CovarianceEstimator})
PortfolioOptimisers.assert_sample_buffer(::PortfolioOptimisers.Online)
PortfolioOptimisers.sample_buffer_seed
PortfolioOptimisers.fold_buffer
PortfolioOptimisers.partial_fit!(state::PortfolioOptimisers.SampleBufferState, X::PortfolioOptimisers.MatNum, F::PortfolioOptimisers.Option{<:PortfolioOptimisers.MatNum} = nothing; dims::Int = 1)
PortfolioOptimisers.assert_buffer_factor_width
PortfolioOptimisers.assert_buffer_presence_agreement
PortfolioOptimisers.reset_empty_buffer
PortfolioOptimisers.seed_sample_buffer
PortfolioOptimisers.seed_buffer_array
PortfolioOptimisers.copy_buffer_rows!
PortfolioOptimisers.reserve_sample_buffer
PortfolioOptimisers.compact_buffer_array
PortfolioOptimisers.partial_fit!(state::PortfolioOptimisers.SampleBufferState, x::PortfolioOptimisers.VecNum, f::PortfolioOptimisers.Option{<:PortfolioOptimisers.VecNum} = nothing)
PortfolioOptimisers.observation_row
PortfolioOptimisers.merge_states(a::PortfolioOptimisers.SampleBufferState, b::PortfolioOptimisers.SampleBufferState)
PortfolioOptimisers.merge_buffer_array
PortfolioOptimisers.trim_merged_array
Base.copy(x::PortfolioOptimisers.SampleBufferState)
PortfolioOptimisers.copy_buffer_array
PortfolioOptimisers.port_opt_view(x::PortfolioOptimisers.SampleBufferState, i, args...)
PortfolioOptimisers.slice_buffer_mask
PortfolioOptimisers.partial_fit!(est::Union{<:PortfolioOptimisers.AbstractEstimator, <:StatsBase.CovarianceEstimator}, X::PortfolioOptimisers.VecNum_MatNum; dims::Int = 1)
PortfolioOptimisers.supports_partial_fit
Online
PortfolioOptimisers.Online_Option
PortfolioOptimisers.Onl
PortfolioOptimisers.online_candidate_fields
PortfolioOptimisers.online_fields
PortfolioOptimisers.online_state_seed(::Union{<:PortfolioOptimisers.AbstractEstimator, <:StatsBase.CovarianceEstimator}, ::PortfolioOptimisers.Option{<:Integer})
PortfolioOptimisers.update_online_estimator
PortfolioOptimisers.estimator_fields
PortfolioOptimisers.online_entry_state
PortfolioOptimisers.online_wrapper_path
PortfolioOptimisers.assert_batch_entry
```
