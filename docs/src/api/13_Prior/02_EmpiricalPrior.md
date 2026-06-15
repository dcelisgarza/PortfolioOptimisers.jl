# Empirical Prior

```@docs
EmpiricalPrior
factory(pe::EmpiricalPrior, w::ObsWeights)
prior(pe::EmpiricalPrior{<:Any, <:Any, Nothing}, X::MatNum, args...; dims::Int = 1,
      kwargs...)
prior(pe::EmpiricalPrior{<:Any, <:Any, <:Number}, X::MatNum, args...;
               dims::Int = 1, kwargs...)
port_opt_view(pr::EmpiricalPrior, rd, args...)
```
