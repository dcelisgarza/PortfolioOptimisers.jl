```@meta
Description = "The programme Allocation Set and the Constrained Update's programme, public API of PortfolioOptimisers.jl: ProgrammeAllocationSet, TsallisProjection, …"
```

# The programme Allocation Set and the Constrained Update's programme

The Allocation Set of the full constraint vocabulary: weight bounds, universe sets, linear constraints, a turnover ceiling against the Price-Adjusted Allocation, a variance or standard-deviation ceiling through the set's own cone, a tracking error and the MIP kinds, with a solver required by its field bound. A projection onto it, and every projection in the Gram geometry, is a bare-model programme assembled by the same builders a JuMP head uses; a programme that does not solve is a Held Step. The two barrier geometries of the first-order rule, the Tsallis and the log-barrier potentials, live here with their scalar roots on the bounded set and their conic programmes on this one.

```@docs
ProgrammeAllocationSet
TsallisProjection
LogBarrierProjection
PortfolioOptimisers.mirror_step
PortfolioOptimisers.set_allocation_set_constraints!
PortfolioOptimisers.rows_needed(set::ProgrammeAllocationSet)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
