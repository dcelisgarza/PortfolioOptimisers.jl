```@meta
Description = "Prior partial fit (b), public API of PortfolioOptimisers.jl: prior, partial_fit!."
```

# Prior partial fit (b)

```@docs
prior(pe::EmpiricalPrior{<:Any, <:Any, Nothing, <:Any, <:Any, <:Option{<:PortfolioOptimisers.PriorCarryState}}; strict::Bool = false, kwargs...)
PortfolioOptimisers.partial_fit!(pe::BlackLittermanPrior, X::PortfolioOptimisers.VecNum_MatNum, F::PortfolioOptimisers.Option{<:PortfolioOptimisers.VecNum_MatNum} = nothing; kwargs...)
prior(pe::BlackLittermanPrior; strict::Bool = false, kwargs...)
```
