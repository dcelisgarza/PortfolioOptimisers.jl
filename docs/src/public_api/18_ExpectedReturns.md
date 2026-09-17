```@meta
Description = "Expected Returns, public API of PortfolioOptimisers.jl: ExpectedReturn, ExpectedReturnRiskRatio, PerformanceSummaryResult, expected_risk, …"
```

# Expected Returns

```@docs
ExpectedReturn
ExpectedReturnRiskRatio
PerformanceSummaryResult
expected_risk(r::ExpectedReturn, w::VecNum, pr::AbstractPriorResult, fees::Option{<:Fees} = nothing; kwargs...)
expected_risk(r::ExpectedReturnRiskRatio, w::VecNum, pr::AbstractPriorResult, fees::Option{<:Fees} = nothing; kwargs...)
performance_summary
expected_risk(r::PrRM, w::VecVecNum, pr::AbstractPriorResult, fees::Option{<:Fees} = nothing; kwargs...)
expected_risk(r::PrRM, pred::PredictionResult{<:Any, <:PredictionReturnsResult{<:Any, <:VecNum}}; kwargs...)
expected_return
expected_ratio
expected_risk_ret_ratio
expected_sric
expected_risk_ret_sric
factory(r::ExpectedReturn, args...; kwargs...)
factory(r::ExpectedReturnRiskRatio, args...; kwargs...)
brinson_attribution
needs_previous_weights(r::ExpectedReturnRiskRatio)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
