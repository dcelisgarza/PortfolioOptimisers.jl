```@meta
Description = "Expected returns, public API of PortfolioOptimisers.jl: ExpectedReturn, ExpectedReturnRiskRatio, expected_risk, expected_return, expected_ratio, …"
```

# Expected returns

```@docs
ExpectedReturn
ExpectedReturnRiskRatio
expected_risk(r::ExpectedReturn, w::VecNum, pr::AbstractPriorResult, fees::Option{<:Fees} = nothing; kwargs...)
expected_risk(r::ExpectedReturnRiskRatio, w::VecNum, pr::AbstractPriorResult, fees::Option{<:Fees} = nothing; kwargs...)
expected_risk(r::PrRM, pred::PredictionResult{<:Any, <:PredictionReturnsResult{<:Any, <:VecNum}}; kwargs...)
expected_return
expected_ratio
expected_risk_ret_ratio
expected_sric
expected_risk_ret_sric
factory(r::ExpectedReturn, args...; kwargs...)
factory(r::ExpectedReturnRiskRatio, args...; kwargs...)
needs_previous_weights(r::ExpectedReturnRiskRatio)
supports_precomputed_returns(::PrRM)
```
