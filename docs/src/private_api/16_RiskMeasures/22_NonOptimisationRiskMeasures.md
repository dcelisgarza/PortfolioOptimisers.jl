```@meta
Description = "Non-Optimisation Risk Measures, private API of PortfolioOptimisers.jl: TCM_Sk, needs_previous_weights, resolve_deferred_quantities, calc_moment_target, …"
```

# Non-Optimisation Risk Measures: private API

```@docs
TCM_Sk
needs_previous_weights(r::MeanReturnRiskRatio)
resolve_deferred_quantities(r::ThirdCentralMoment, pr::AbstractPriorResult)
calc_moment_target(::TCM_Sk{Nothing, Nothing}, ::Any, x::VecNum)
calc_deviations_vec(r::TCM_Sk, w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
calc_deviations_vec(r::TCM_Sk, x::VecNum)
supports_precomputed_returns(r::ThirdCentralMoment)
```
