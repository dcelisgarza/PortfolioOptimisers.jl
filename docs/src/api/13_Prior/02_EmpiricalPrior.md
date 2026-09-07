# Empirical Prior

```@docs
EmpiricalPrior
prior(pe::EmpiricalPrior{<:Any, <:Any, Nothing}, X::MatNum,
      F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing;
      dims::Int = 1, kwargs...)
prior(pe::EmpiricalPrior{<:Any, <:Any, <:Number}, X::MatNum,
      F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing;
      dims::Int = 1, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
