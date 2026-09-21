```@meta
Description = "The programme of a programme Allocation Set, public API of PortfolioOptimisers.jl: set_allocation_set_constraints!."
```

# The programme of a programme Allocation Set

The builders of a programme Allocation Set's model on both arms — the bare projection model of the Constrained Update, and the Allocation Set Constraint a follow-the-leader rule appends to its held optimiser — in the order a JuMP head's builders run, the second-stage resolution on the head's rows, the projection objectives in every geometry with the set's penalties folded in, and the Held Step of a programme that does not solve.

```@docs
PortfolioOptimisers.set_allocation_set_constraints!
```
