```@meta
Description = "Custom value expected returns, public API of PortfolioOptimisers.jl: CustomExpectedReturnsValueAlgorithm, CustomValueExpectedReturns, mean, port_opt_view."
```

# Custom value expected returns

```@docs
CustomExpectedReturnsValueAlgorithm
CustomValueExpectedReturns
mean(me::CustomValueExpectedReturns{<:Number}, X::MatNum; dims::Int = 1, kwargs...)
mean(me::CustomValueExpectedReturns{<:VecNum}, X::MatNum; dims::Int = 1, kwargs...)
mean(me::CustomValueExpectedReturns{<:Union{<:Function, <:CustomExpectedReturnsValueAlgorithm}}, X::MatNum; dims::Int = 1, kwargs...)
port_opt_view(me::CustomValueExpectedReturns{<:VecNum}, i, args...)
```
