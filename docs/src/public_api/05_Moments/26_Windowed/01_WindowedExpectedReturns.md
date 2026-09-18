```@meta
Description = "Windowed expected returns, public API of PortfolioOptimisers.jl: WindowedExpectedReturns, factory, mean."
```

# Windowed expected returns

```@docs
WindowedExpectedReturns
factory(ce::WindowedExpectedReturns, args...; kwargs...)
mean(me::WindowedExpectedReturns, X::MatNum; dims::Int = 1, iv::Option{<:MatNum} = nothing, kwargs...)
```
