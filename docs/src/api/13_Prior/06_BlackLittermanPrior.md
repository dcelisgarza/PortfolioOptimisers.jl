# Black-Litterman Prior

```@docs
BlackLittermanPrior
Base.getproperty(obj::BlackLittermanPrior, sym::Symbol)
prior(pe::BlackLittermanPrior, X::MatNum,
               F::Option{<:MatNum} = nothing; dims::Int = 1,
               strict::Bool = false, kwargs...)
calc_omega
vanilla_posteriors
remove_excl_views
```
