# Black-Litterman Prior

```@docs
BlackLittermanPrior
prior(pe::BlackLittermanPrior, X::MatNum,
               F::Option{<:MatNum} = nothing; dims::Int = 1,
               strict::Bool = false, kwargs...)
calc_omega
bl_preroll
vanilla_posteriors
apply_rf
remove_excl_views
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
