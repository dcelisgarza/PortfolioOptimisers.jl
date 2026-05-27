# Black-Litterman Prior

```@docs
BlackLittermanPrior
factory(pe::BlackLittermanPrior, w::ObsWeights)
Base.getproperty(obj::BlackLittermanPrior, sym::Symbol)
prior(pe::BlackLittermanPrior, X::MatNum,
               F::Option{<:MatNum} = nothing; dims::Int = 1,
               strict::Bool = false, kwargs...)
prior_view(pr::BlackLittermanPrior, rd)
calc_omega
vanilla_posteriors
remove_excl_views
```
