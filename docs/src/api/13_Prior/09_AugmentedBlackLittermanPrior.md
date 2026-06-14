# Augmented Black-Litterman Prior

```@docs
AugmentedBlackLittermanPrior
factory(pe::AugmentedBlackLittermanPrior, w::ObsWeights)
Base.getproperty(obj::AugmentedBlackLittermanPrior, sym::Symbol)
prior(pe::AugmentedBlackLittermanPrior, X::MatNum, F::MatNum;
               dims::Int = 1, strict::Bool = false, kwargs...)
port_opt_view(pr::AugmentedBlackLittermanPrior, rd, args...)
```
