```@meta
Description = "Moment Risk Measures, public API of PortfolioOptimisers.jl: FirstLowerMoment, MeanAbsoluteDeviation, SecondMoment, EvenMoment, LowOrderMoment, …"
```

# Moment Risk Measures

```@docs
FirstLowerMoment
MeanAbsoluteDeviation
SecondMoment
EvenMoment
LowOrderMoment
ThirdLowerMoment
FourthMoment
StandardisedHighOrderMoment
HighOrderMoment
factory(alg::StandardisedHighOrderMoment, w::ObsWeights)
factory(alg::MomentMeasureAlgorithm, args...; kwargs...)
factory(r::LowOrderMoment, pr::AbstractPriorResult, args...; kwargs...)
port_opt_view(r::LowOrderMoment, i, args...)
factory(r::HighOrderMoment, pr::AbstractPriorResult, args...; kwargs...)
port_opt_view(r::HighOrderMoment, i, args...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
