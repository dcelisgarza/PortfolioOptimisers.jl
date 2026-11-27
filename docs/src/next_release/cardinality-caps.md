```@meta
Description = "The cardinality caps take a CountOf or a FractionOf, and a fraction resolves against the set that the fit sees."
```

# [Cardinality caps as a count or a fraction](@id migration-next-cardinality-caps)

The cardinality caps `card` and `scard` of `ConstraintBlock`, `JuMPOptimiser` and `ProgrammeAllocationSet`, and the `B` of `IntegerPhylogenyEstimator` and `IntegerPhylogeny`, take a `CountOf` or a `FractionOf`. A fraction resolves against the set that the fit sees, so `card = FractionOf(0.2)` keeps its meaning when a view or a resample reduces the universe, or when an asset leaves it. A plain `Integer` keeps its meaning, and the slot holds the same integer. A fraction rounds down, to at least one member.

| Call | The previous release | This release |
| --- | --- | --- |
| `JuMPOptimiser(; card = 0.2)`, and a plain real number between zero and one in `scard` or in `B` | A `TypeError` | Reads `FractionOf(0.2)`. |
| `JuMPOptimiser(; card = 3.0)` or `card = 1.0`, and the same in `scard` and `B` | A `TypeError` | Raises a `DomainError`. Write `3`, `CountOf(3)` or `FractionOf(1)`. |
| `IntegerPhylogeny(; A, B).B` with a `FractionOf` in `B` | A `TypeError` | A vector of integers, one cap for each row of the stored `A`. |
| A projection onto a `ProgrammeAllocationSet` that reads no rows, in a direct call of `project`, with a precomputed `lcse` or `gcarde` whose width is not the number of assets | The model raised the error | Raises the `DimensionMismatch` of the constraint width, as the same set raises it on rows. |
| `JuMPOptimiser(; smtx = [A, B], …)` with precomputed matrices and no `sets` | Raised a `MethodError` at the fit | Solves, as one precomputed matrix does. |
