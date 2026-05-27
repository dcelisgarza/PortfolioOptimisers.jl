# Kurtosis

```@docs
Kurtosis
factory(r::Kurtosis, pr::HighOrderPrior, args...; kwargs...)
factory(r::Kurtosis, pr::LowOrderPrior, args...; kwargs...)
calc_moment_target(::Kurtosis{<:Any, Nothing, Nothing, <:Any, <:Any, <:Any, <:Any}, ::Any, x::VecNum)
calc_deviations_vec(r::Kurtosis, w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
```
