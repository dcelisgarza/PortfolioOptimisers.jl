```@meta
Description = "Base regression, public API of PortfolioOptimisers.jl: AbstractTimeSeriesRegressionEstimator, AbstractCrossSectionalRegressionEstimator, …"
```

# [Base regression](@id api-base-regression)

```@docs
AbstractTimeSeriesRegressionEstimator
AbstractCrossSectionalRegressionEstimator
AbstractRegressionTarget
factory(tgt::AbstractRegressionTarget, w::ObsWeights)
regression_target_weights(tgt::AbstractRegressionTarget)
regression_target_weights(tgt::Union{LinearModel, GeneralisedLinearModel})
is_basis_invariant
LinearModel
GeneralisedLinearModel
Regression
factory(re::LinearModel, w::ObsWeights)
StatsAPI.fit(tgt::LinearModel, X::MatNum, y::VecNum)
factory(re::GeneralisedLinearModel, w::ObsWeights)
StatsAPI.fit(tgt::GeneralisedLinearModel, X::MatNum, y::VecNum)
regression(re::Regression, args...)
regression(re::AbstractTimeSeriesRegressionEstimator, rd::ReturnsResult)
port_opt_view(re::Regression, i, args...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
