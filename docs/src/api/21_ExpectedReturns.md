# Expected Returns

## Public

```@docs
ExpectedReturn
ExpectedReturnRiskRatio
expected_risk(r::ExpectedReturn, w::VecNum, pr::AbstractPriorResult,
                       fees::Option{<:Fees} = nothing; kwargs...)
expected_risk(r::ExpectedReturnRiskRatio, w::VecNum, pr::AbstractPriorResult,
                       fees::Option{<:Fees} = nothing; kwargs...)
```

## Private

```@docs
PerfRM
PrRM
expected_return
expected_ratio
expected_risk_ret_ratio
expected_sric
expected_risk_ret_sric
factory(r::ExpectedReturn, args...; kwargs...)
factory(r::ExpectedReturnRiskRatio, args...; kwargs...)
needs_previous_weights(r::ExpectedReturnRiskRatio)
brinson_attribution
```
