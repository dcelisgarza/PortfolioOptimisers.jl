# Implied Volatility

```@docs
ImpliedVolatilityAlgorithm
ImpliedVolatilityRegression
ImpliedVolatilityPremium
ImpliedVolatility
realised_vol
implied_vol
predict_realised_vols(::ImpliedVolatilityPremium, iv::MatNum, ::Any, ivpa::Nothing)
predict_realised_vols(::ImpliedVolatilityPremium, iv::MatNum, ::Any,
                               ivpa::Num_VecNum)
predict_realised_vols(alg::ImpliedVolatilityRegression, iv::MatNum, X::MatNum,
                               ::Any)
cov(ce::ImpliedVolatility, X::MatNum; dims::Int = 1, mean = nothing,
                        iv::MatNum, ivpa::Option{<:Num_VecNum} = nothing, kwargs...)
cor(ce::ImpliedVolatility, X::MatNum; dims::Int = 1, mean = nothing,
                        iv::MatNum, ivpa::Option{<:Num_VecNum} = nothing, kwargs...)
```
