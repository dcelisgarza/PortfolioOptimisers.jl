```@meta
Description = "Propagatable, public API of PortfolioOptimisers.jl: @propagatable, @fprop, @vprop, @pprop, @wprop, @cprop, factory."
```

# Propagatable

## Utility functions

```@docs
@propagatable
@fprop
@vprop
@pprop
@wprop
@cprop
```

## Summary statistics

Some estimators and constraints reduce a vector to one number, such as its minimum, mean, median or maximum. Each type below names one reduction, and some of them take observation weights. `vec_to_real_measure` applies the reduction to a vector.

```@docs
factory(mv::MeanValue, args...; kwargs...)
factory(mdv::MedianValue, args...; kwargs...)
factory(sv::StdValue, args...; kwargs...)
factory(vv::VarValue, args...; kwargs...)
factory(msv::StandardisedValue, args...; kwargs...)
```
