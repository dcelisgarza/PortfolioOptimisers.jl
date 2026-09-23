```@meta
Description = "Gerber covariance, private API of PortfolioOptimisers.jl: gerber_updown, concordance_counts, gerber."
```

# Gerber covariance: private API

The Gerber statistic measures how often two assets move together. A movement smaller than a threshold does not count. The statistic counts the observations at which both assets move beyond their thresholds in the same direction, and those at which they move beyond them in opposite directions. It uses the counts and not the sizes of the movements, so an extreme movement weighs as much as any other. It extends Kendall's tau [gerber](@cite).

The library implements the three published variants, `Gerber0`, `Gerber1` and `Gerber2`, which normalise the counts in different ways [gerber_analysis](@cite).

## The three variants

The functions below count the joint movements and compute the statistic of each variant.

```@docs
gerber_updown
concordance_counts
gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber0}, X::MatNum, sd::ArrNum)
gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber1}, X::MatNum, sd::ArrNum)
gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber2}, X::MatNum, sd::ArrNum)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
