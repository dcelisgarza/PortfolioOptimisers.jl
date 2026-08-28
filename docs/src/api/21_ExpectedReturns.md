# Expected Returns

## Public

```@docs
ExpectedReturn
ExpectedReturnRiskRatio
expected_risk(r::ExpectedReturn, w::VecNum, pr::AbstractPriorResult,
                       fees::Option{<:Fees} = nothing; kwargs...)
expected_risk(r::ExpectedReturnRiskRatio, w::VecNum, pr::AbstractPriorResult,
                       fees::Option{<:Fees} = nothing; kwargs...)
PerformanceSummaryResult
performance_summary
```

## Private

```@docs
PerfRM
PrRM
expected_risk(r::PrRM, w::VecVecNum, pr::AbstractPriorResult,
                       fees::Option{<:Fees} = nothing; kwargs...)
expected_risk(r::PrRM,
                       pred::PredictionResult{<:Any,
                                              <:PredictionReturnsResult{<:Any, <:VecNum}};
                       kwargs...)
expected_return
term_fees
expected_ratio
expected_risk_ret_ratio
sric_penalty
expected_sric
expected_risk_ret_sric
factory(r::ExpectedReturn, args...; kwargs...)
factory(r::ExpectedReturnRiskRatio, args...; kwargs...)
needs_previous_weights(r::ExpectedReturnRiskRatio)
prrm_prediction_message
brinson_attribution
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
