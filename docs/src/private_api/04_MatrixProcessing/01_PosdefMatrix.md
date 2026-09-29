```@meta
Description = "Posdef matrix, private API of PortfolioOptimisers.jl: posdef_accepts, posdef_repair!, clipped_correlation, clipped_repair_holds, …"
```

# Posdef matrix: private API

`posdef_accepts` and `posdef_repair!` are the two steps of [`posdef!`](@ref) that each algorithm of [`Posdef`](@ref) owns: the test that leaves a matrix unchanged, and the repair. `clipped_correlation`, `clipped_repair_holds` and `higham_alternating_projections` are the steps of [`ClippedNearestCorrelation`](@ref).

`zero_variance_rows!` is the helper of [`posdef!`](@ref) and [`matrix_processing_block!`](@ref) for a constant variable. It sets the row and column of each zero variance to zero, and returns the block of positive variances that the repair works on.

```@docs
PortfolioOptimisers.posdef_accepts
PortfolioOptimisers.posdef_repair!
PortfolioOptimisers.clipped_correlation
PortfolioOptimisers.clipped_repair_holds
PortfolioOptimisers.higham_alternating_projections
PortfolioOptimisers.zero_variance_rows!
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
