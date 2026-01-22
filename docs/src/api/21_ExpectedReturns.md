# Expected Returns

## Public

```@docs
ReturnRiskMeasure
ReturnRiskRatioRiskMeasure
expected_risk(r::ReturnRiskMeasure, w::VecNum, pr::AbstractPriorResult,
                       fees::Option{<:Fees} = nothing; kwargs...)
expected_risk(r::ReturnRiskRatioRiskMeasure, w::VecNum, pr::AbstractPriorResult,
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
factory(r::ReturnRiskRatioRiskMeasure, pr::AbstractPriorResult, args...; kwargs...)
factory(r::ReturnRiskRatioRiskMeasure{<:Any, <:UncertaintySetVariance}, ucs::UcSE_UcS;
                 kwargs...)
factory(r::ReturnRiskRatioRiskMeasure{<:Any, <:SlvRM}, slv::Slv_VecSlv; kwargs...)
factory(r::ReturnRiskRatioRiskMeasure{<:Any, <:TnTrRM}, w::VecNum)
brinson_attribution
```
