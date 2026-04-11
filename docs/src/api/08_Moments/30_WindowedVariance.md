# Windowed variance

```@docs
WindowedVariance
factory(ce::WindowedVariance, w::ObsWeights)
var(ce::WindowedVariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
var(ce::WindowedVariance, X::VecNum; mean = nothing)
std(ce::WindowedVariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
std(ce::WindowedVariance, X::VecNum; mean = nothing)
```
