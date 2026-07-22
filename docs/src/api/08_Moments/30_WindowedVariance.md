# Windowed variance

```@docs
WindowedVariance
var(ve::WindowedVariance, X::MatNum; dims::Int = 1, mean = nothing, iv::Option{<:MatNum} = nothing, kwargs...)
var(ve::WindowedVariance, X::VecNum; mean = nothing)
std(ve::WindowedVariance, X::MatNum; dims::Int = 1, mean = nothing, iv::Option{<:MatNum} = nothing, kwargs...)
std(ve::WindowedVariance, X::VecNum; mean = nothing)
```
