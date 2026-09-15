# Exponentially Weighted Covariance

## Types

```@docs
ExpWeightedCovariance
ExpWeightedCovarianceState
process_observation!(cache::ExpWeightedCovarianceState, ce::ExpWeightedCovariance,
                              X::VecNum, active_mask::Option{<:AbstractVector{<:Bool}})
exp_weighted_pass!(f, est::ExpWeightedCovariance, X::MatNum, dims::Int,
                            active_mask::Option{<:AbstractMatrix{<:Bool}},
                            state::Option{<:ExpWeightedCovarianceState} = nothing)
exp_weighted_pass!(est::ExpWeightedCovariance, X::MatNum, dims::Int,
                            active_mask::Option{<:AbstractMatrix{<:Bool}},
                            state::Option{<:ExpWeightedCovarianceState} = nothing)
exp_weighted_moment(cache::ExpWeightedCovarianceState, est::ExpWeightedCovariance)
cov(ce::ExpWeightedCovariance, X::MatNum; dims::Int = 1,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
gap_fill_value(::ExpWeightedCovariance)
cor(ce::ExpWeightedCovariance, X::MatNum; dims::Int = 1,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
cov(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel};
                        dims::Int = 1, kwargs...)
cor(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel};
                        dims::Int = 1, kwargs...)
var(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel};
                        dims::Int = 1, kwargs...)
std(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel};
                        dims::Int = 1, kwargs...)
variance_series(ce::ExpWeightedCovariance, X::MatNum; dims::Int = 1,
                         active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
variance_series(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel};
                         dims::Int = 1, kwargs...)
partial_fit!(ce::ExpWeightedCovariance, X::MatNum; dims::Int = 1,
                      active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
partial_fit!(ce::ExpWeightedCovariance, x::VecNum;
                      active_mask::Option{<:AbstractVector{<:Bool}} = nothing, kwargs...)
cov(ce::ExpWeightedCovariance, state::ExpWeightedCovarianceState;
                        kwargs...)
cov(ce::ExpWeightedCovariance; kwargs...)
cor(ce::ExpWeightedCovariance, state::ExpWeightedCovarianceState;
                        kwargs...)
cor(ce::ExpWeightedCovariance; kwargs...)
PortfolioOptimisers.merge_states(a::ExpWeightedCovarianceState, b::ExpWeightedCovarianceState)
Base.copy(x::PortfolioOptimisers.ExpWeightedCovarianceState)
```
