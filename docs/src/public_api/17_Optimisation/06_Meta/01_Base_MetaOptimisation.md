```@meta
Description = "Meta optimisation, public API of PortfolioOptimisers.jl: SubPortfolioUniverse, sub_portfolio_count, sub_portfolio_predict, sub_portfolio_view, …"
```

# Meta optimisation

A meta-optimiser solves one inner problem for each sub-portfolio, and gives the outer optimiser a synthetic universe with one asset for each sub-portfolio. [`NestedClustered`](@ref) enumerates cluster index sets, and [`Stacking`](@ref) enumerates inner optimisers. A new enumeration subtypes `SubPortfolioUniverse` and adds a method of `sub_portfolio_count`, `sub_portfolio_predict`, `sub_portfolio_view` and `fold_weight_matrix`.

```@docs
PortfolioOptimisers.SubPortfolioUniverse
PortfolioOptimisers.sub_portfolio_count
PortfolioOptimisers.sub_portfolio_predict
PortfolioOptimisers.sub_portfolio_view
PortfolioOptimisers.fold_weight_matrix
```
