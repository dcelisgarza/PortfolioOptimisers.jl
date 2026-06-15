# Bayesian Black-Litterman Prior

```@docs
BayesianBlackLittermanPrior
factory(pe::BayesianBlackLittermanPrior, w::ObsWeights)
Base.getproperty(obj::BayesianBlackLittermanPrior, sym::Symbol)
prior(pe::BayesianBlackLittermanPrior, X::MatNum, F::MatNum;
               dims::Int = 1, strict::Bool = false, kwargs...)
port_opt_view(pr::BayesianBlackLittermanPrior, rd, args...)
```
