```@meta
Description = "The online portfolio selection head, public API of PortfolioOptimisers.jl: OnlinePortfolioSelection, optimise, partial_fit!, factory, port_opt_view, …"
```

# The online portfolio selection head

The one head of the family: a naive optimiser whose batch verb is the Causal Pass over every row of its carrier, whose `partial_fit!` is the Block Step, and whose `optimise(opt)` with no data is the Recursion Read-out.

The rules the head runs sit on the pages of this group by the set they were built in; by Li and Hoi's (2014) five families — benchmarks, follow the winner, follow the loser, pattern matching and meta-learning — they are listed in the [capability catalogue](@ref catalogue-online-portfolio-selection) and in the user guide's [roster by group](@ref user-guide-online-selection-roster), which also runs the family on two synthetic markets and tabulates its regret.

```@docs
OnlinePortfolioSelection
optimise(opt::OnlinePortfolioSelection{<:Any, <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult; dims::Int = 1, kwargs...)
optimise(opt::OnlinePortfolioSelection; kwargs...)
PortfolioOptimisers.partial_fit!(opt::OnlinePortfolioSelection{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:PortfolioOptimisers.Option{<:PortfolioOptimisers.OnlinePortfolioSelectionState}}, rd::ReturnsResult)
factory(opt::OnlinePortfolioSelection, w::VecNum)
PortfolioOptimisers.port_opt_view(opt::OnlinePortfolioSelection, i, args...)
_optimise(opt::OnlinePortfolioSelection, rd::ReturnsResult; dims::Int = 1, kwargs...)
PortfolioOptimisers.rows_needed(opt::OnlinePortfolioSelection)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
