# High Order Factor Prior

```@docs
AbstractHighOrderPriorEstimator_F
AbstractHiLoOrderPriorEstimator_F
HighOrderFactorPriorEstimator
factory(pe::HighOrderFactorPriorEstimator, w::ObsWeights)
coskewness_residuals
cokurtosis_residuals
Base.getproperty(obj::HighOrderFactorPriorEstimator, sym::Symbol)
prior(pe::HighOrderFactorPriorEstimator, X::MatNum, F::MatNum; dims::Int = 1, kwargs...)
port_opt_view(pr::HighOrderFactorPriorEstimator, rd)
```
