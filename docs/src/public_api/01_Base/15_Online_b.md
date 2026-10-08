```@meta
Description = "Online (b), public API of PortfolioOptimisers.jl: supports_partial_fit, AbstractAllocationSet, AbstractProgrammeAllocationSet."
```

# Online (b)

A family that folds states it through `supports_partial_fit`, and an outer estimator asks each of its members before it folds them.

```@docs
PortfolioOptimisers.supports_partial_fit
```

The page also has the supertypes of the allocation sets. An allocation set is the set of allowed weights that an online portfolio selection rule projects its step onto.

```@docs
PortfolioOptimisers.AbstractAllocationSet
PortfolioOptimisers.AbstractProgrammeAllocationSet
```
