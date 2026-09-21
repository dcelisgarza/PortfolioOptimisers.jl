```@meta
Description = "The programme Allocation Set and the Constrained Update's programme, public API of PortfolioOptimisers.jl: ProgrammeAllocationSet, TsallisProjection, …"
```

# The programme Allocation Set and the Constrained Update's programme

The Allocation Set of the full constraint vocabulary: every constraint kind a `JuMPOptimiser` takes, under the optimiser's field names and type bounds, the optimiser's direct objective penalties, and a solver required by its field bound. Its name-keyed slots resolve once per fold and its row-reading slots at every step; the programme itself, on both arms, is the next file's. The two barrier geometries of the first-order rule, the Tsallis and the log-barrier potentials, live here with their scalar roots on the bounded set.

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
