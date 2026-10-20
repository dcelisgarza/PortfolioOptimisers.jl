```@meta
Description = "Windowed Cokurtosis, public API of PortfolioOptimisers.jl: WindowedCokurtosis, cokurtosis."
```

# [Windowed Cokurtosis](@id api-windowed-cokurtosis)

```@docs
WindowedCokurtosis
cokurtosis(kte::WindowedCokurtosis, X::MatNum; dims::Int = 1, mean = nothing, iv::Option{<:MatNum} = nothing, kwargs...)
cokurtosis(kte::WindowedCokurtosis, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, mean = nothing, kwargs...)
```
