```@meta
Description = "Regression, public API of PortfolioOptimisers.jl: LinearModel, GeneralisedLinearModel, Regression, factory, StatsAPI.fit, regression, port_opt_view."
```

# [Regression](@id api-regression)

```@docs
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
