```@meta
Description = "JuMP model optimisation, public API of PortfolioOptimisers.jl: Solver, JuMPResult."
```

# JuMP model optimisation

`PortfolioOptimisers.jl` builds its optimisation models with [`JuMP`](https://github.com/jump-dev/JuMP.jl), so a solver that JuMP supports can solve them.

A `Solver` names one solver, its settings and the solution statuses it must reach. An optimiser takes one `Solver` or a vector of them, and tries them in order until one returns an accepted solution. A `JuMPResult` records which solvers failed, at which stage, and whether any of them succeeded.

```@docs
Solver
JuMPResult
```
