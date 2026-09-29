```@meta
Description = "Posdef matrix, private API of PortfolioOptimisers.jl: zero_variance_rows!."
```

# Posdef matrix: private API

This is the helper of [`posdef!`](@ref) and [`matrix_processing_block!`](@ref) for a constant variable. `zero_variance_rows!` sets the row and column of each zero variance to zero, and returns the block of positive variances that the repair works on.

```@docs
PortfolioOptimisers.zero_variance_rows!
```
