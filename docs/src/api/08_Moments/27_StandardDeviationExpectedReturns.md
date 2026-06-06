# Standard deviation expected returns

```@docs
StandardDeviationExpectedReturns
factory(ce::StandardDeviationExpectedReturns, args...; kwargs...)
mean(me::StandardDeviationExpectedReturns, X::MatNum;
                         dims::Int = 1, kwargs...)
moment_view(me::StandardDeviationExpectedReturns, i)
VarianceExpectedReturns
factory(ce::VarianceExpectedReturns, args...; kwargs...)
mean(me::VarianceExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
moment_view(me::VarianceExpectedReturns, i)
```
