# JuMP model optimisation

`PortfolioOptimisers.jl` is based on [`JuMP`](https://github.com/jump-dev/JuMP.jl), as such it tries to be as flexible as possible.

Theese types and functions let us define solver and solution interfaces.

```@docs
DictStrA_VecPairStrA
SlvSettings
Solver
VecSlv
Slv_VecSlv
AbstractJuMPResult
JuMPResult
set_solver_attributes
optimise_JuMP_model!
```
