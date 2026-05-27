# High Order Factor Prior

```@docs
AbstractHighOrderPriorEstimator_F
AbstractHiLoOrderPriorEstimator_F
HighOrderFactorPriorEstimator
factory(pe::HighOrderFactorPriorEstimator, w::ObsWeights)
coskewness_residuals(X::MatNum, me::AbstractExpectedReturnsEstimator)
cokurtosis_residuals(sigma::MatNum, X::MatNum, me::AbstractExpectedReturnsEstimator, ex::FLoops.Transducers.Executor)
Base.getproperty(obj::HighOrderFactorPriorEstimator, sym::Symbol)
prior(pe::HighOrderFactorPriorEstimator, X::MatNum, F::MatNum; dims::Int = 1, kwargs...)
prior_view(pr::HighOrderFactorPriorEstimator, rd)
```
