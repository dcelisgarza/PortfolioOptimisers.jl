```@meta
Description = "Elastic net, public API of PortfolioOptimisers.jl: AbstractPenaltyStrength, penalty_strength, penalty_strength_gradient, MiddleOfPath, FixedStrength, …"
```

# Elastic net

[`ElasticNet`](@ref) is a penalised regression target. It fits a least-squares regression with a penalty on the ``L_1`` norm and on the squared ``L_2`` norm of the coefficients, and every estimator that takes a regression target can fit it. The strength of the penalty comes from a rule: [`MiddleOfPath`](@ref) reads it off the data at the middle of the regularisation path, and [`FixedStrength`](@ref) takes it from you. A new rule subtypes `AbstractPenaltyStrength` and adds a method of `penalty_strength`.

The fit states its effective degrees of freedom, so [`StepwiseRegression`](@ref) can score it under `:aic`, `:aicc` and `:bic`. [`StrengthDivergence`](@ref) adds the term of a strength that reads the data to that count.

## Strength rules

```@docs
PortfolioOptimisers.AbstractPenaltyStrength
PortfolioOptimisers.penalty_strength
PortfolioOptimisers.penalty_strength_gradient
MiddleOfPath
FixedStrength
```

## Strength divergence

```@docs
PortfolioOptimisers.AbstractStrengthDivergence
StrengthDivergence
PortfolioOptimisers.strength_divergence
```

## Target

```@docs
ElasticNet
factory(tgt::ElasticNet, w::ObsWeights)
factory(tgt::ElasticNet, ic::InterceptColumn)
PortfolioOptimisers.regression_target_weights(tgt::ElasticNet)
PortfolioOptimisers.StatsAPI.fit(tgt::ElasticNet, X::MatNum, y::VecNum)
PortfolioOptimisers.StatsAPI.predict(m::PortfolioOptimisers.ElasticNetFit, X::MatNum)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
