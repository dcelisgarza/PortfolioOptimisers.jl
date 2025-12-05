# Expected Returns

## Public

```@docs
ReturnRiskMeasure
RatioRiskMeasure
expected_risk(r::ReturnRiskMeasure, w::VecNum, pr::AbstractPriorResult,
                       fees::Option{<:Fees} = nothing; kwargs...)
expected_risk(r::RatioRiskMeasure, w::VecNum, pr::AbstractPriorResult,
                       fees::Option{<:Fees} = nothing; kwargs...)
```

## Private

```@docs
expected_return
expected_ratio
expected_risk_ret_ratio
expected_sric
expected_risk_ret_sric
factory(r::ReturnRiskMeasure, pr::AbstractPriorResult, args...; kwargs...)
factory(r::ReturnRiskMeasure, args...; kwargs...)
factory(r::RatioRiskMeasure, pr::AbstractPriorResult, args...; kwargs...)
factory(r::RatioRiskMeasure{<:Any, <:UncertaintySetVariance}, ucs::UcSE_UcS;
                 kwargs...)
factory(r::RatioRiskMeasure{<:Any, <:SlvRM}, slv::Slv_VecSlv; kwargs...)
factory(r::RatioRiskMeasure{<:Any, <:TnTrRM}, w::VecNum)
brinson_attribution
```
