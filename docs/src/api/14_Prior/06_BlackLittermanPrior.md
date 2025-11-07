# Black-Litterman Prior

```@docs
BlackLittermanPrior
prior(pe::BlackLittermanPrior, X::NumMat,
               F::Union{Nothing, <:NumMat} = nothing; dims::Int = 1,
               strict::Bool = false, kwargs...)
PortfolioOptimisers.calc_omega
PortfolioOptimisers.vanilla_posteriors
```
