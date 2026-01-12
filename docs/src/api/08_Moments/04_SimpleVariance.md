# Variance and standard deviation

The variance is used throughout the library, it can be used as part of the expected return and covariance estimation as well as in performance analysis and constraint generation. It is trivial to compute the standard deviation from the variance, so we provide those too.

```@docs
SimpleVariance
std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
std(ve::SimpleVariance, X::VecNum; dims::Int = 1, mean = nothing, kwargs...)
var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
var(ve::SimpleVariance, X::VecNum; mean = nothing)
```
