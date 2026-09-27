```@meta
Description = "Gerber covariance, public API of PortfolioOptimisers.jl: BaseGerberCovariance, GerberCovarianceAlgorithm, Gerber0, Gerber1, Gerber2, GerberCovariance, cov, …"
```

# Gerber covariance

The Gerber statistic measures how often two assets move together. An observation counts only when a return moves past a threshold, and it counts as one vote whatever the size of the move. Small moves, which are mostly noise, do not count, and one extreme move cannot dominate the result. The statistic extends Kendall's tau. It compares the votes where the two assets move past their thresholds in the same direction with the votes where they move in opposite directions [gerber](@cite).

Three variants are published, and the library has all three, `Gerber0`, `Gerber1` and `Gerber2`. `Gerber0` and `Gerber1` divide the net vote by different counts, and `Gerber2` normalises the whole matrix, so the three give different estimates from the same data [gerber_analysis](@cite).

## Abstract Gerber covariance types

Subtype these to write a new Gerber covariance estimator or algorithm.

```@docs
BaseGerberCovariance
GerberCovarianceAlgorithm
```

## Gerber covariance estimators and algorithms

This section has the three variants of the statistic, and the estimator that takes one of them.

```@docs
Gerber0
Gerber1
Gerber2
GerberCovariance
cov(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any}, X::MatNum; dims::Int = 1, kwargs...)
cor(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any}, X::MatNum; dims::Int = 1, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
