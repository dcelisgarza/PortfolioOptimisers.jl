# Factor Prior

```@docs
FactorPrior
factory(pe::FactorPrior, w::ObsWeights)
Base.getproperty(obj::FactorPrior, sym::Symbol)
prior(pe::FactorPrior, X::MatNum, F::MatNum; dims::Int = 1,
               kwargs...)
port_opt_view(pr::FactorPrior, rd, args...)
```
