```@meta
Description = "Online (b), public API of PortfolioOptimisers.jl: AbstractWindowRule, RollingWindow, SeedWindow, AbstractChoiceRule, BatchChoice, PinnedChoice, …"
```

# Online (b)

A window rule says which observations the window of a windowed moment estimator keeps on the online step. A choice rule says how a choice that a fit makes over its whole sample behaves on the online step.

```@docs
PortfolioOptimisers.AbstractWindowRule
RollingWindow
SeedWindow
PortfolioOptimisers.AbstractChoiceRule
BatchChoice
PinnedChoice
```

A family that folds states it through `supports_partial_fit`, and an outer estimator asks each of its members before it folds them.

```@docs
PortfolioOptimisers.supports_partial_fit
```

The page also has the supertypes of the allocation sets. An allocation set is the set of allowed weights that an online portfolio selection rule projects its step onto.

```@docs
PortfolioOptimisers.AbstractAllocationSet
PortfolioOptimisers.AbstractProgrammeAllocationSet
```
