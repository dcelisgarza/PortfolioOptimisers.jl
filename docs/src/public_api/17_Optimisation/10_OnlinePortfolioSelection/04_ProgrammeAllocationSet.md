```@meta
Description = "The programme Allocation Set and the Constrained Update's programme, public API of PortfolioOptimisers.jl: ProgrammeAllocationSet, TsallisProjection, …"
```

# The programme Allocation Set and the Constrained Update's programme

`ProgrammeAllocationSet` is a set of allowed weights that takes every constraint a `JuMPOptimiser` takes, with the same field names and types, and the objective penalties of a `JuMPOptimiser`. A projection onto it is an optimisation, so it needs a solver. The set resolves the constraints that name assets once per fold, and rebuilds the constraints that read returns at every update. This page also has the two barrier geometries of the first-order rule, `TsallisProjection` and `LogBarrierProjection`. Their projection onto a `BoundedAllocationSet` is a root search on one scalar.

```@docs
ProgrammeAllocationSet
TsallisProjection
LogBarrierProjection
PortfolioOptimisers.mirror_step
PortfolioOptimisers.rows_needed(set::ProgrammeAllocationSet)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
