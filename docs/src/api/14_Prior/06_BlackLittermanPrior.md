# Black-Litterman Prior

```@docs
BlackLittermanPrior
prior(pe::BlackLittermanPrior, X::AbstractMatrix,
               F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1,
               strict::Bool = false, kwargs...)
PortfolioOptimisers.calc_omega
PortfolioOptimisers.vanilla_posteriors
```
