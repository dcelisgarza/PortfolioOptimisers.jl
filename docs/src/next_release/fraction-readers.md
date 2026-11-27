```@meta
Description = "A count that a reader takes from a fraction compares the share of the count with the fraction, so a floating-point product does not move it by one."
```

# [Counts that a fraction gives](@id migration-next-fraction-readers)

Some readers of a fraction rounded the floating-point product of the fraction and the size, so the count could be one away from the count that the fraction states. They now compare the share of the count with the fraction. The default fractions and significance levels keep their counts, except in the row for the positions of `sbar = nothing`. There the sum of the prior probabilities rounded below `alpha`, and the window had two losses too many.

| Reader | Case | The previous release | This release |
| --- | --- | --- | --- |
| `NearestNeighbourMatch(; neighbours = FractionOf(0.29))` | 100 candidates | 28 neighbours | 29 |
| `IntegerConditionalValueatRiskView(; sbar = FractionOf(0.07))` | 100 observations | 8 losses | 7 |
| `IntegerConditionalValueatRiskView(; sbar = nothing)`, the floor of `2 * alpha * T` | `alpha = 0.07`, 100 observations | 15 losses | 14 |
| `IntegerConditionalValueatRiskView(; sbar = nothing)`, the positions at which the prior probabilities reach `alpha` | `alpha = 0.05`, 120 observations with equal probabilities | 14 losses | 12 |
| `RankRule(; best = FractionOf(0.575))`, and the same in `worst` | 100 assets | 57 assets | 58 |
| `LowerTailDependenceCovariance(; alpha = 0.07)` | 100 observations | 8 observations | 7 |
| `HillTailDecay` and `RadialTailDecay` at `alpha = 0.07` | 100 pooled values or distances | 8 values | 7 |
| `MCDCovariance(; alpha = 0.55, shr = MCDShrinkage())` | 100 observations | 56 observations | 55 |
| `MCDCovariance(; alpha = 0.58)` | 101 observations of one asset | 58 observations | 59 |
