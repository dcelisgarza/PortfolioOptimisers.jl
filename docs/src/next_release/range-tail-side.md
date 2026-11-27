```@meta
Description = "A range builds each tail on its own side: the results that move, and what a range or a measure of your own declares to be a gain tail."
```

# [A range builds each tail on its own side](@id migration-next-range-tail-side)

- **A range of your own whose tail is a range, or an `OrderedWeightsArray` or `ValueatRiskRange` as its gain tail,** builds each tail on its own side. The previous release built both tails of an inner range on one side, and it built the loss side of an OWA measure with `ExactOrderedWeightsArray` or `SortingNetworkOrderedWeightsArray`, and of a `ValueatRiskRange` with `DistributionValueatRisk`. The model then optimised a value that was not the value of the range.
- **A range of your own with an `UncertaintySetVariance` as a tail** takes an estimator as the `ucs` of that tail. The previous release raised a `MethodError`.
- **A `Frontier` on the gain tail of a range of your own** sweeps a tail that is a `LowOrderMoment` with one value on both sides, a range that reads the weights, or a `ValueatRiskRange` with `DistributionValueatRisk`. The previous release raised an `ArgumentError` for these tails, as [its text](@ref migration-next-frontier-range-tail-extending) states. Every other tail that reads the weights, the returns and the fees still raises it, and its message now names these tails.

## [If you extend the library](@id migration-next-range-tail-side-extending)

- **The gain tail of a `PortfolioOptimisers.RangeTails`** must be a measure that `PortfolioOptimisers.supports_gain_tail` answers `true` for. The constructor refuses any other with an `ArgumentError`, such as a drawdown measure, a `LowOrderMoment` that reads one side of the deviations or a target that holds a scalar, `NegativeSkewness`, a tracking measure or `TurnoverRiskMeasure`, whose builder built the loss tail a second time in the previous release. A measure of your own whose `set_risk_constraints!` reads the keyword `loss` declares `PortfolioOptimisers.supports_gain_tail(::MyRisk) = true`, and so does a range of your own that fuses its tails.
- **`PortfolioOptimisers.gain_tail_risk` is public.** It gives the value of a tail on the negated portfolio, which a `Frontier` on the gain tail reads at its corners. A fee keeps its sign when the weights change sign, so the library has no generic value of a measure that reads the weights, the returns and the fees. A measure of your own of that kind, also a range of your own that fuses its tails, declares `PortfolioOptimisers.gain_tail_risk(r::MyRisk, w, X, fees, ::PortfolioOptimisers.WeightsReturnsFeesInput)` to be the gain tail of a range with a `Frontier`. The error of `range_tails` for a range that fuses names this method.
