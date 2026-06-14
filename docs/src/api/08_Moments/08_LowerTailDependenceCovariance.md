# Lower Tail Dependence Covariance

```@docs
LowerTailDependenceCovariance
factory(ce::LowerTailDependenceCovariance, w::ObsWeights)
cov(ce::LowerTailDependenceCovariance, X::MatNum; dims::Int = 1, kwargs...)
cor(ce::LowerTailDependenceCovariance, X::MatNum; dims::Int = 1, kwargs...)
port_opt_view(ce::LowerTailDependenceCovariance, i, args...)
lower_tail_dependence
```
