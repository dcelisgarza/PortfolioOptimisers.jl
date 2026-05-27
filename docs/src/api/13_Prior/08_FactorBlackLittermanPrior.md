# Factor Black-Litterman Prior

```@docs
FactorBlackLittermanPrior
factory(pe::FactorBlackLittermanPrior, w::ObsWeights)
Base.getproperty(obj::FactorBlackLittermanPrior, sym::Symbol)
prior(pe::FactorBlackLittermanPrior, X::MatNum, F::MatNum;
               dims::Int = 1, strict::Bool = false, kwargs...)
prior_view(pr::FactorBlackLittermanPrior, rd)
```
