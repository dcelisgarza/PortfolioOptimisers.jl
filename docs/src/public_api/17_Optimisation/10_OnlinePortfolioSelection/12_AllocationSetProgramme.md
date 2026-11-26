```@meta
Description = "Allocation Set programme, public API of PortfolioOptimisers.jl: set_allocation_set_constraints!, project, mask_reads_returns."
```

# Allocation Set programme

`set_allocation_set_constraints!` adds the constraints of a `ProgrammeAllocationSet` to a JuMP model, in the order a JuMP optimiser adds them. The model is the projection of a raw step onto the set of allowed weights. A follow-the-leader rule adds the same constraints to the optimisation it solves again at each period, through a private function that registers them under other names, so the rule keeps its own constraints and gains those of the set. When the projection does not solve, the rule keeps the weights it had after the price moves of the period, so it trades nothing in that period.

`mask_reads_returns` answers whether the investable mask of a prior reads only the returns and the active mask of the panel. A restart of a first-order rule on a `RiskLoss` then skips the fit of the prior on rows with no gap. A moment estimator of a user that reads another input adds a method that returns `false`.

```@docs
PortfolioOptimisers.set_allocation_set_constraints!
PortfolioOptimisers.project(::PortfolioOptimisers.AbstractProjectionGeometry, ::PortfolioOptimisers.AbstractAllocationSet, ::AbstractVector, ::AbstractVector, ::Nothing)
PortfolioOptimisers.project(::PortfolioOptimisers.AbstractProjectionGeometry, ::Tuple, ::AbstractVector, ::AbstractVector, ::PortfolioOptimisers.Option{<:PortfolioOptimisers.ReturnsResult})
PortfolioOptimisers.mask_reads_returns
```
