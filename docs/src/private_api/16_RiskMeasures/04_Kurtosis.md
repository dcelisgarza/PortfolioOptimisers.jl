```@meta
Description = "Kurtosis, private API of PortfolioOptimisers.jl: calc_moment_target, calc_deviations_vec."
```

# Kurtosis: private API

```@docs
calc_moment_target(::Kurtosis{<:Any, Nothing, Nothing, <:Any, <:Any, <:Any, <:Any}, ::Any, x::VecNum)
calc_deviations_vec(r::Kurtosis, w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
calc_deviations_vec(r::Kurtosis, x::VecNum)
```
