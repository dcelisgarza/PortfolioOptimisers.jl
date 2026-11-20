```@meta
Description = "Non-Optimisation Risk Measures, private API of PortfolioOptimisers.jl: resolve_deferred_quantities, calc_moment_target, calc_deviations_vec, reader_frames."
```

# Non-Optimisation Risk Measures: private API

```@docs
resolve_deferred_quantities(r::ThirdCentralMoment, pr::AbstractPriorResult)
calc_moment_target(::ThirdCentralMoment{<:Any, Nothing, Nothing}, ::Any, x::VecNum)
calc_deviations_vec(r::ThirdCentralMoment, w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
calc_deviations_vec(r::ThirdCentralMoment, x::VecNum)
PortfolioOptimisers.reader_frames(::MeanReturnRiskRatio, ::AbstractString, ::Bool)
```
