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
expected_return
expected_ratio
expected_risk_ret_ratio
expected_sric
expected_risk_ret_sric
factory(r::ExpectedReturn, pr::AbstractPriorResult, args...; kwargs...)
factory(r::ExpectedReturn, args...; kwargs...)
factory(r::ExpectedReturnRiskRatio, pr::AbstractPriorResult, args...; kwargs...)
factory(r::ExpectedReturnRiskRatio{<:Any, <:UncertaintySetVariance}, ucs::UcSE_UcS;
                 kwargs...)
factory(r::ExpectedReturnRiskRatio{<:Any, <:SlvRM}, slv::Slv_VecSlv; kwargs...)
factory(r::ExpectedReturnRiskRatio{<:Any, <:TnTrRM}, w::VecNum)
brinson_attribution
```
