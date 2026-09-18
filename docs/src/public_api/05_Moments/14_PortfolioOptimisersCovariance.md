```@meta
Description = "PortfolioOptimisersCovariance, public API of PortfolioOptimisers.jl: PortfolioOptimisersCovariance, cov, cor, partial_fit!."
```

# PortfolioOptimisersCovariance

```@docs
PortfolioOptimisersCovariance
cov(ce::PortfolioOptimisersCovariance, X::MatNum; dims = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
cor(ce::PortfolioOptimisersCovariance, X::MatNum; dims = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
cov(ce::PortfolioOptimisersCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims = 1, kwargs...)
PortfolioOptimisers.partial_fit!(ce::PortfolioOptimisersCovariance{<:Any, <:Any, Nothing}, X::MatNum; dims::Int = 1, kwargs...)
cov(ce::PortfolioOptimisersCovariance{<:Any, <:Any, Nothing}; kwargs...)
cov(ce::PortfolioOptimisersCovariance{<:Any, <:Any, <:PortfolioOptimisers.SampleBufferState})
```
