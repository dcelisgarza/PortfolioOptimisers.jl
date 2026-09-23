```@meta
Description = "The programme of a programme Allocation Set, public API of PortfolioOptimisers.jl: set_allocation_set_constraints!."
```

# The programme of a programme Allocation Set

`set_allocation_set_constraints!` adds the constraints of a `ProgrammeAllocationSet` to a JuMP model, in the order a JuMP optimiser adds them. It works on two kinds of model. The first is the projection of a raw step onto the set of allowed weights. The second is the optimisation that a follow-the-leader rule solves again at each period, which then keeps its own constraints and gains those of the set. When the projection does not solve, the rule keeps the weights it had after the price moves of the period, so it trades nothing in that period.

```@docs
PortfolioOptimisers.set_allocation_set_constraints!
```
