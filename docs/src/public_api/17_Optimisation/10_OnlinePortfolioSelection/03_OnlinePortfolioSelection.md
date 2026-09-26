```@meta
Description = "The online portfolio selection head, public API of PortfolioOptimisers.jl: OnlinePortfolioSelection, optimise, partial_fit!, factory, port_opt_view, Online, …"
```

# The online portfolio selection head

`OnlinePortfolioSelection` runs an online portfolio selection rule. It is a naive optimiser, so it needs a solver only when its set of allowed weights does. `optimise(opt, rd)` runs the rule from the start weights over every row of the returns in order, and returns the weights for the next period. `partial_fit!` updates the rule with a block of new rows, one row at a time. `optimise(opt)` with no data returns the current weights of the rule, and runs no batch fit.

The pages of this group list the rules in the order the library added them. Li and Hoi (2014) sort the rules into five families: benchmarks, follow the winner, follow the loser, pattern matching and meta-learning. The [capability catalogue](@ref catalogue-online-portfolio-selection) and the [roster by group](@ref user-guide-online-selection-roster) of the user guide list the rules by these families. The user guide also runs the rules on two synthetic markets and prints a table of their regret.

```@docs
OnlinePortfolioSelection
optimise(opt::OnlinePortfolioSelection{<:Any, <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult; dims::Int = 1, kwargs...)
optimise(opt::OnlinePortfolioSelection; kwargs...)
PortfolioOptimisers.partial_fit!(opt::OnlinePortfolioSelection{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:PortfolioOptimisers.Option{<:PortfolioOptimisers.OnlinePortfolioSelectionState}}, rd::ReturnsResult)
factory(opt::OnlinePortfolioSelection, w::VecNum)
PortfolioOptimisers.port_opt_view(opt::OnlinePortfolioSelection, i, args...)
Online(::OnlinePortfolioSelection, args...)
_optimise(opt::OnlinePortfolioSelection, rd::ReturnsResult; dims::Int = 1, kwargs...)
PortfolioOptimisers.rows_needed(opt::OnlinePortfolioSelection)
PortfolioOptimisers.rows_needed(td::TimeDependent)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
