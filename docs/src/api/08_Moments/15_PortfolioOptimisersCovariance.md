# PortfolioOptimisersCovariance

```@docs
PortfolioOptimisers.find_uncorrelated_indices
PortfolioOptimisersCovariance
cov(ce::PortfolioOptimisersCovariance, X::MatNum; dims = 1,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
cor(ce::PortfolioOptimisersCovariance, X::MatNum; dims = 1,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
gap_fill_value(::PortfolioOptimisersCovariance)
cov(ce::PortfolioOptimisersCovariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims = 1, kwargs...)
```
