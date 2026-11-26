```@meta
Description = "Migration: a Frontier on one tail of a range takes the span of that tail, in MeanRisk and in NearOptimalCentering."
```

# [A `Frontier` on one tail of a range](@id migration-next-frontier-range-tail)

- **A `Frontier` in the `ub` of one tail of a `GenericValueatRiskRange`** takes its levels from the value of that tail at the two corner portfolios, in `MeanRisk` and in `NearOptimalCentering`. Before this change it took them from the value of the whole range, so the levels were too high and most points of the sweep were one portfolio. A range of your own, a subtype of `RangeRiskMeasure` that the generic builder builds, takes the same levels.

## [If you extend the library](@id migration-next-frontier-range-tail-extending)

- **A `Frontier` on the gain tail of a range of your own** raises an `ArgumentError` when that tail reads the weights, the returns and the fees, a `WeightsReturnsFeesInput` measure, except a `ValueatRisk` with `DistributionValueatRisk`. Such a tail has no value on the negated portfolio, so the sweep cannot value its corners. Before this change the sweep took the span of the whole range.
